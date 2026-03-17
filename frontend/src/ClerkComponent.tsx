import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, {
  useEffect,
  useRef,
  ReactElement,
} from "react"
import { SignIn, SignUp } from "@clerk/clerk-react"
import { useUser, useClerk } from "@clerk/clerk-react"

const COMPONENT_URL = import.meta.env.VITE_COMPONENT_HOST_URL

/**
 * Handles the logic for a custom Clerk Component, which controls user login UIUX.
 * Arguments are passed from Streamlit to this component to determine which action to take.
 * - action='startup' checks for an already logged-in user for when the app is booted up.
 * - action='logout' logs out the user.
 * - action='login' displays the Clerk SignIn widget and sets Streamlit.setComponentValue(user).
 *
 * @param {ComponentProps} props - The props object passed from Streamlit
 * @param {Object} props.args - Custom arguments passed from the Python side
 * @param {Object} props.theme - Streamlit theme object for consistent styling
 * @returns {ReactElement} The rendered component
 */
function ClerkComponent({ args, theme }: ComponentProps): ReactElement {

  // Extract custom arguments passed from Python
  const action = args["action"]
  const heightOffset = args["height_offset"]
  const heightMinimum = args["min_height"]
  const visible = args["visible"]

  const { isSignedIn, user } = useUser()
  const { signOut } = useClerk()
  const params = new URLSearchParams(window.location.search);
  const mode = params.get("mode") == "signup" ? "signup" : "signin";

  const containerRef = useRef(null)
  const startupTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // check for a resize
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const height = entry.contentRect.height + heightOffset > heightMinimum ? entry.contentRect.height + heightOffset : heightMinimum;
        if (visible) {
          Streamlit.setFrameHeight(height);
        } else {
          Streamlit.setFrameHeight(0);
        }
      }
    });

    observer.observe(element);

  }, []);

  if (action == "logout") {
    signOut()
    return (
      <span>
        LOGOUT. &nbsp;
      </span>
    )
  }

  useEffect(() => {

    // If no user data appears after a full second, assume the user is logged out.
    if (action == "startup" && !(isSignedIn && user)) {
      if (!startupTimeoutRef.current) {
        startupTimeoutRef.current = setTimeout(() => {
          Streamlit.setComponentValue("NO_USER");
        }, 1000);
      }
    }

    if (isSignedIn && user) {
      // Clear any pending startup timeout once we have a signed-in user.
      if (startupTimeoutRef.current) {
        clearTimeout(startupTimeoutRef.current);
        startupTimeoutRef.current = null;
      }

      // send user data to Streamlit
      const safeUser = {
        signedIn: isSignedIn,
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.primaryEmailAddress?.emailAddress,
        createdAt: user.createdAt
      }
      Streamlit.setComponentValue(safeUser);
    }

    // Cleanup: clear any pending startup timeout on effect cleanup.
    return () => {
      if (startupTimeoutRef.current) {
        clearTimeout(startupTimeoutRef.current);
        startupTimeoutRef.current = null;
      }
    }
  }, [isSignedIn, user, action]);

  if (!isSignedIn) {
    if (mode == "signin") {
      return (
        <div ref={containerRef}>
          <SignIn 
            routing="virtual"
            forceRedirectUrl={COMPONENT_URL}
            signUpForceRedirectUrl={COMPONENT_URL}
            signUpUrl={`${COMPONENT_URL}?mode=signup`}
          />
        </div>
      )
    } else if (mode == "signup") {
      return (
        <div ref={containerRef}>
          <SignUp
            routing="virtual"
            forceRedirectUrl={COMPONENT_URL}
            signInForceRedirectUrl={COMPONENT_URL}
            oauthFlow="popup"
            signInUrl={`${COMPONENT_URL}?mode=signin`}
          />
        </div>
      )
    }
  }
  return (
    <span>
      Loading... &nbsp;
    </span>
  )
}

/**
 * withStreamlitConnection is a higher-order component (HOC) that:
 * 1. Establishes communication between this component and Streamlit
 * 2. Passes Streamlit's theme settings to your component
 * 3. Handles passing arguments from Python to your component
 * 4. Handles component re-renders when Python args change
 *
 * You don't need to modify this wrapper unless you need custom connection behavior.
 */
export default withStreamlitConnection(ClerkComponent)