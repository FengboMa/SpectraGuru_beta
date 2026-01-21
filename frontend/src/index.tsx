import React, { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import ClerkComponent from "./ClerkComponent"
import { ClerkProvider } from "@clerk/clerk-react"

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

const rootElement = document.getElementById("root")

if (!rootElement) {
  console.log("Root element not found")
  throw new Error("Root element not found")
}

const root = createRoot(rootElement)

root.render(
  <StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
      <ClerkComponent />
    </ClerkProvider>
  </StrictMode>
)