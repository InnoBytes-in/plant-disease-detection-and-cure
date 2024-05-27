import { useAuth } from "@clerk/clerk-react"
import Header from "../components/shared/Header"
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";
import ProductList from "../components/ProductList";

const Products = () => {
  const {userId, isLoaded} = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if(!userId && isLoaded) {
      navigate('/sign-in')
    }
  }, [userId, navigate, isLoaded]);

  if (!isLoaded) return "Loading...";

  return (
    <div>
      <Header />
      <div className="container mt-8">
        <div className="justify-between items-center mx-16">
            <h1 className="font-bold text-3xl">Products</h1>

            <ProductList />
        </div>
      </div>
    </div>
  )
}

export default Products
