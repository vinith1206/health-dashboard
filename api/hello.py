# Minimal Python serverless function to verify Python runtime on Vercel.
# Vercel will route requests to /api/hello — this file is used to validate the platform.

def handler(request):
    return ("Hello from Python on Vercel", 200)