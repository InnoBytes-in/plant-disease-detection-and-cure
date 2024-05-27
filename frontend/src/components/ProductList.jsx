/* eslint-disable react/prop-types */
import { useEffect, useState } from "react";
import ProductCard from "../components/ProductCard";
import { fetch } from "../api";

const ProductList = () => {
    const [products, setProducts] = useState([]);
    const [diseases, setDiseases] = useState([]);
    const [plants, setPlants] = useState([]);
    const [selectedDisease, setSelectedDisease] = useState('');
    const [selectedPlant, setSelectedPlant] = useState('');
    // const [keyword, setKeyword] = useState('');
  
    useEffect(() => {
        const fetchProducts = async () => {
          try {
            const response = await fetch();
            setProducts(response.data);
          } catch (error) {
            console.error('Error fetching products:', error);
          }
        };
        fetchProducts();
      }, []);
    
      useEffect(() => {
        // Extracting unique diseases and plants
        const allDiseases = new Set();
        const allPlants = new Set();
        products.forEach(product => {
          product.targetDisease.forEach(disease => allDiseases.add(disease));
          product.targetPlant.forEach(plant => allPlants.add(plant));
        });
        setDiseases([...allDiseases]);
        setPlants([...allPlants]);
      }, [products]);
    
      const handleDiseaseChange = (e) => {
        setSelectedDisease(e.target.value);
      };
    
      const handlePlantChange = (e) => {
        setSelectedPlant(e.target.value);
      };
    
      const filteredProducts = products.filter(product => {
        return (selectedDisease ? product.targetDisease.includes(selectedDisease) : true) &&
               (selectedPlant ? product.targetPlant.includes(selectedPlant) : true);
      });
  
    return (
      <div>
            <div className="flex justify-center gap-5 mt-5">
              <select onChange={handleDiseaseChange} className="bg-[#317b30] border-none text-white px-5 py-3 text-sm cursor-pointer rounded-xl" value={selectedDisease}>
                <option value="" className="font-semibold">Filter by Disease</option>
                {diseases.map((disease, index) => (
                  <option key={index} value={disease}>{disease}</option>
                ))}
              </select>
              <select onChange={handlePlantChange} className="bg-[#317b30] border-none text-white px-5 py-3 text-sm cursor-pointer rounded-xl w-auto" value={selectedPlant}>
                <option value="" className="font-semibold">Filter by Plant</option>
                {plants.map((plant, index) => (
                  <option key={index} value={plant}>{plant}</option>
                ))}
              </select>
              {/* <input type='text' name='q' onChange={(e) => setKeyword(e.target.value)} placeholder='Search Products...' className='mx-5' /> */}
            </div>
            <div className="grid sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-5 pt-8 gap-5">
                {filteredProducts.map((product) => 
                    <ProductCard
                        key={product._id}
                        img={product.image}
                        name={product.name}
                        price={product.price} 
                    />
                )}
            </div>
      </div>
    );
  };


export default ProductList;