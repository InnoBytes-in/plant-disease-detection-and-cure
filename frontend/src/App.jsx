import { ClerkProvider } from "@clerk/clerk-react";
import { Outlet, useNavigate } from "react-router-dom";

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

if (!PUBLISHABLE_KEY) {
  throw new Error("Missing Publishable Key")
}

const App = () => {
  const navigate = useNavigate();

  return (
    <ClerkProvider navigate={navigate} publishableKey={PUBLISHABLE_KEY} appearance={{variables: {colorPrimary: "#2E8B57"}}}>
      <main>
        <Outlet />
      </main>
    </ClerkProvider>
  )
}

export default App
