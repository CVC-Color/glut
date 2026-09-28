# Re-export the compiled kernel entry point so callers can use `glut_cuda.forward(...)`.
from glut_cuda.glut_cuda import forward  # noqa: F401
