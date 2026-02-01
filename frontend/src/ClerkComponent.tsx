import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  ReactElement,
} from "react"
import { SignIn } from "@clerk/clerk-react"
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
  const height = args["height"]

  const { isSignedIn, user } = useUser()
  const { signOut } = useClerk()

  useEffect(() => {
    // Call this when the component's size might change
    Streamlit.setFrameHeight(height)
    // Adding the style and theme as dependencies since they might
    // affect the visual size of the component.
  }, [theme])

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
      let timeoutId: ReturnType<typeof setTimeout>;
      timeoutId = setTimeout(() => {
        Streamlit.setComponentValue("NO_USER");
      }, 1000);
    }

    if (isSignedIn && user) {
      // Send the user object to Streamlit
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
  }, [isSignedIn, user]);

  if (!isSignedIn) {
    return (
      <span>
        <SignIn 
          routing="virtual"
          forceRedirectUrl={COMPONENT_URL}
        />
      </span>
    )
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