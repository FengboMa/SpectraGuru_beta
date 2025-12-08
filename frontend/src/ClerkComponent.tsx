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
import { useUser } from "@clerk/clerk-react"

/**
 * A template for creating Streamlit components with React
 *
 * This component demonstrates the essential structure and patterns for
 * creating interactive Streamlit components, including:
 * - Accessing props and args sent from Python
 * - Managing component state with React hooks
 * - Communicating back to Streamlit via Streamlit.setComponentValue()
 * - Using the Streamlit theme for styling
 * - Setting frame height for proper rendering
 *
 * @param {ComponentProps} props - The props object passed from Streamlit
 * @param {Object} props.args - Custom arguments passed from the Python side
 * @param {Object} props.theme - Streamlit theme object for consistent styling
 * @returns {ReactElement} The rendered component
 */
function ClerkComponent({ args, theme }: ComponentProps): ReactElement {
  // Extract custom arguments passed from Python
  const height = args["height"]

  const { isSignedIn, user } = useUser()

  useEffect(() => {
    // Call this when the component's size might change
    Streamlit.setFrameHeight(height)
    // Adding the style and theme as dependencies since they might
    // affect the visual size of the component.
  }, [theme])

  useEffect(() => {
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
        Hello, there! &nbsp;
        <SignIn />
      </span>
    )
  }

  return (
    <span>
      Already signed in. &nbsp;
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