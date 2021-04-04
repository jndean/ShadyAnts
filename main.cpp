

#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>



extern "C" void step_simulation(unsigned int *g_odata, int imgw, int imgh);
extern "C" void CUDAinit(unsigned int** texture, int imgw, int imgh, int ubyte_size);

unsigned int image_width = 1920;
unsigned int image_height = 1080;
unsigned int window_width = image_width;
unsigned int window_height = image_height;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

unsigned int *cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;
struct cudaGraphicsResource *cuda_tex_screen_resource;


GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result


// Timer
#define REFRESH_DELAY     0 //ms
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

GLuint shDraw;

// GL functionality
bool initGL(int *argc, char **argv);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void reshape(int w, int h);

static const char *glsl_drawtex_vertshader_src =
    "void main(void)\n"
    "{\n"
    "	gl_Position = gl_Vertex;\n"
    "	gl_TexCoord[0].xy = gl_MultiTexCoord0.xy;\n"
    "}\n";

static const char *glsl_drawtex_fragshader_src =
    "#version 130\n"
    "uniform usampler2D texImage;\n"
    "void main()\n"
    "{\n"
    "   vec4 c = texture(texImage, gl_TexCoord[0].xy);\n"
    "	gl_FragColor = c / 255.0;\n"
    "}\n";

static const char *glsl_draw_fragshader_src =
    "#version 130\n"
    "out uvec4 FragColor;\n"
    "void main()\n"
    "{"
    "  FragColor = uvec4(gl_Color.xyz * 255.0, 255.0);\n"
    "}\n";


void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    // The texture is a 8 bits UI, scale the fetch with a GLSL shader
    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"
    SDK_CHECK_ERROR_GL();

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

    glUseProgram(0);
    
    SDK_CHECK_ERROR_GL();
}


void display()
{
    sdkStartTimer(&timer);

    step_simulation(cuda_dest_resource, image_width, image_height);

    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));
    int size_tex_data = image_width * image_height * 4 * sizeof(GLubyte) ;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
    displayImage(tex_cudaResult);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // flip backbuffer
    glutSwapBuffers();

    // Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit)
    {
        char cTitle[256];
        float fps = 1000.0f / sdkGetAverageTimerValue(&timer);
        sprintf(cTitle, "CUDA GL Post Processing (%d x %d): %.1f fps", window_width, window_height, fps);
        glutSetWindowTitle(cTitle);
        fpsCount = 0;
        fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        sdkResetTimer(&timer);
    }
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    if (key == 27) {
	// if ESC, cleanup and exit
	sdkDeleteTimer(&timer);
	cudaFree(cuda_dest_resource);
	glDeleteTextures(1, &tex_screen);
	SDK_CHECK_ERROR_GL();
	tex_screen = 0;
	glDeleteTextures(1, &tex_cudaResult);
	SDK_CHECK_ERROR_GL();
	tex_cudaResult = 0;
	if (iGLUTWindowHandle) {
	    glutDestroyWindow(iGLUTWindowHandle);
	}
	exit(EXIT_SUCCESS);
    }
}

void reshape(int w, int h)
{
    window_width = w;
    window_height = h;
}



int main(int argc, char **argv)
{
    char *Xstatus = getenv("DISPLAY");
    if (Xstatus == NULL) {
        printf("X server is not running\n");
        exit(EXIT_WAIVED);
    }
    setenv ("DISPLAY", ":0", 0);

    if (false == initGL(&argc, argv)) {
        return EXIT_FAILURE;
    }
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);


    CUDAinit(&cuda_dest_resource, image_width, image_height, sizeof(GLubyte));

    glutMainLoop();

    return EXIT_SUCCESS;
}



GLuint compileGLSLprogram(const char *vertex_shader_src, const char *fragment_shader_src)
{
    GLuint v, f, p = 0;

    p = glCreateProgram();

    if (vertex_shader_src)
    {
        v = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(v, 1, &vertex_shader_src, NULL);
        glCompileShader(v);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            char temp[256] = "";
            glGetShaderInfoLog(v, 256, NULL, temp);
            printf("Vtx Compile failed:\n%s\n", temp);
            glDeleteShader(v);
            return 0;
        }
        else
        {
            glAttachShader(p,v);
        }
    }

    if (fragment_shader_src)
    {
        f = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(f, 1, &fragment_shader_src, NULL);
        glCompileShader(f);

        // check if shader compiled
        GLint compiled = 0;
        glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);

        if (!compiled)
        {
            char temp[256] = "";
            glGetShaderInfoLog(f, 256, NULL, temp);
            printf("frag Compile failed:\n%s\n", temp);
            glDeleteShader(f);
            return 0;
        }
        else
        {
            glAttachShader(p,f);
        }
    }

    glLinkProgram(p);

    int infologLength = 0;
    int charsWritten  = 0;

    glGetProgramiv(p, GL_INFO_LOG_LENGTH, (GLint *)&infologLength);

    if (infologLength > 0)
    {
        char *infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(p, infologLength, (GLsizei *)&charsWritten, infoLog);
        printf("Shader compilation error: %s\n", infoLog);
        free(infoLog);
    }

    return p;
}



bool initGL(int *argc, char **argv)
{
    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("CUDA OpenGL post-processing");

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported (2,0) ||
        !areGLExtensionsSupported (
            "GL_ARB_pixel_buffer_object "
            "GL_EXT_framebuffer_object"
        ))
    {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1f, 10.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glEnable(GL_LIGHT0);
    float red[] = { 1.0f, 0.1f, 0.1f, 1.0f };
    float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 60.0f);

    SDK_CHECK_ERROR_GL();

    // create texture that will receive the result of CUDA
    glGenTextures(1, &tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, image_width, image_height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_cudaResult,
                                                GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));


    
    // load shader programs
    shDraw = compileGLSLprogram(NULL, glsl_draw_fragshader_src);

    shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
    SDK_CHECK_ERROR_GL();
    
    return true;
}
