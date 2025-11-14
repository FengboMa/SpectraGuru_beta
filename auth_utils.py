# auth_utils.py
import os, requests, urllib.parse
from dotenv import load_dotenv

load_dotenv()                      # reads .env in local dev

CLERK_JWKS_KEY = os.getenv("CLERK_JWKS_KEY")
CLERK_PUBLIC_KEY = os.getenv("CLERK_PUBLISHABLE_KEY")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
FRONTEND_API    = os.getenv("CLERK_FRONTEND_API")
APP_URL         = os.getenv("APP_URL")          # where Clerk should bounce back

def clerk_signin_url() -> str:
    app_url = os.getenv("APP_URL") or "http://localhost:80"   # fallback → never None
    redirect = urllib.parse.quote_plus(app_url)
    return (
        f"https://{FRONTEND_API}/sign-in?"
        f"redirect_url={redirect}"
        "&create_session=true"          # ← NEW
    )

def verify_clerk_session(token: str):
    if not token:
        return None

    # ① try universal verify  ────────────────────────────────
    resp = requests.post(
        f"https://api.clerk.com/v1/token/verify",
        headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
        json={"token": token},
        timeout=5,
    )
    print("XXX",resp.text)
    print("XXX")
    print("TRACE  /tokens/verify →", resp.status_code, resp.text[:150])

    if resp.status_code == 200:
        claims = resp.json().get("claims", {})
        return {
            "id":         claims.get("sub"),
            "first_name": claims.get("given_name", "User"),
            "last_name":  claims.get("family_name", ""),
            "email":      claims.get("email"),
        }

    # ② fallback: treat as session-id  ───────────────────────
    resp = requests.get(
        f"https://api.clerk.com/v1/sessions/{token}",
        headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
        timeout=5,
    )
    print("TRACE  /sessions/id   →", resp.status_code, resp.text[:150])

    if resp.status_code == 200 and resp.json().get("status") == "active":
        data = resp.json()
        return {
            "id": data["user_id"],
            "first_name": "User",
            "last_name":  "",
            "email":      None,
        }
    print("DEBUG  CLERK_SECRET_KEY =", bool(CLERK_SECRET_KEY))

    return None
