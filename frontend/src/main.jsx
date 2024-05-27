import React from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider, createBrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './index.css'

import About from './pages/About'
import CheckDisease from './pages/CheckDisease'
import Contact from './pages/Contact'
import Home from './pages/Home'
import Products from './pages/Products'
import SignInPage from './pages/SignInPage.jsx'
import SignUpPage from './pages/SignUpPage.jsx'


  const router = createBrowserRouter([
    {
      element: <App />,
      children: [
        {
          path: '/',
          element: <Home />
        },
        {
          path: '/about',
          element: <About />
        },
        {
          path: '/contact',
          element: <Contact />
        },
        {
          path: '/check-disease',
          element: <CheckDisease />
        },
        {
          path: '/sign-in', element: <SignInPage />
        },
        {
          path: 'sign-up', element: <SignUpPage />
        },
        {
          path: '/products', element: <Products />
        },
      ]
    }
  ])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
