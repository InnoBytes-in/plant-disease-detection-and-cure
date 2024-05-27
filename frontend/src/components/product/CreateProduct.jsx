import { useState } from "react";
import { createProduct } from "../../api";

const CreateProduct = () => {
    const [productData, setProductData] = useState({
        name: '', chemicalCompound: '', manufacturer: '', targetDisease: '', targetPlant: '', price: 0, countInStock: 0, rating: 0, image: ''
    });
    
    const handleSubmit = (e) => {
        e.preventDefault();

        createProduct()
    }

  return (
    <div className="container">
        <h1>Create Product</h1>
        <form autoComplete="off" noValidate onSubmit={handleSubmit}>
            <input type="text" name="name" value={productData.name} onChange={(e) => setProductData({ ...productData, name: e.target.value })} />
            <input type="text" name="manufacturer" value={productData.name} onChange={(e) => setProductData({ ...productData, name: e.target.value })} />
            <input type="text" name="name" value={productData.name} onChange={(e) => setProductData({ ...productData, name: e.target.value })} />
            <input type="text" name="name" value={productData.name} onChange={(e) => setProductData({ ...productData, name: e.target.value })} />
            <input type="text" name="name" value={productData.name} onChange={(e) => setProductData({ ...productData, name: e.target.value })} />
            <input type="text" name="name" value={productData.name} onChange={(e) => setProductData({ ...productData, name: e.target.value })} />
        </form>
    </div>
  )
}

export default CreateProduct
