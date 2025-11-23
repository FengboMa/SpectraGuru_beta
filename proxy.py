from fastapi import FastAPI, Request, HTTPException
import requests
from jose import jwt
import os, requests
from dotenv import load_dotenv

load_dotenv("CLERK_TEST.env")                      # reads .env in local dev

CLERK_PUBLIC_KEY = os.getenv("CLERK_TEST_PUBLISHABLE_KEY")
CLERK_SECRET_KEY = os.getenv("CLERK_TEST_SECRET_KEY")
FRONTEND_API    = os.getenv("CLERK_TEST_FAPI")
ACCOUNT_PORTAL = os.getenv("CLERK_TEST_ACCOUNT_PORTAL")
JWKS_URL = os.getenv("CLERK_TEST_JWKS_URL")

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#JWKS_URL = "https://<your-clerk-domain>/.well-known/jwks.json"
#CLERK_ISSUER = "https://<your-clerk-domain>"
#CLERK_AUDIENCE = "<your-clerk-audience>"

def verify_clerk_token(token: str):
    jwks = requests.get(JWKS_URL).json()
    print("JWKS:",jwks)
    """
    try:
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=CLERK_AUDIENCE,
            issuer=CLERK_ISSUER
        )
        return claims
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    """

#@app.middleware("http")
async def clerk_auth_middleware(request: Request, call_next):
    print("MIDDLEWARE")
    # First try cookie
    token = request.cookies.get("__session")

    # Fallback: Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split("Bearer ")[1]

    if not token:
        raise HTTPException(status_code=401, detail="Missing Clerk token")

    claims = verify_clerk_token(token)
    request.state.user = claims
    return await call_next(request)


@app.get("/user")
async def get_user(request: Request):
    # Query params
    print("Query params:", dict(request.query_params))

    # Headers
    print("Headers:", dict(request.headers))

    # Cookies
    print("Cookies:", request.cookies)

    from clerk_backend_api import Clerk
    from clerk_backend_api.security import authenticate_request
    from clerk_backend_api.security.types import AuthenticateRequestOptions

    sdk = Clerk(bearer_auth=f"{CLERK_SECRET_KEY}")
    request_state = sdk.authenticate_request(
        request,
        AuthenticateRequestOptions(
            authorized_parties=['http://localhost', 'http://localhost:8000', 'http://localhost:8501']
        )
    )
    print("PAYLOAD:",request_state)


    #return {"user": request.state.user}

