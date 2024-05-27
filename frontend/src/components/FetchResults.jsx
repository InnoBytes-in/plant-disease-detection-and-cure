import { useState, useEffect } from 'react';
import io from 'socket.io-client';
import { fetch } from "../api";
import ProductCard from './ProductCard';

const socket = io('https://streamlit-node-connector.onrender.com');

function FetchResults() {
    const [products, setProducts] = useState([]);
  const [filteredProducts, setFilteredProducts] = useState([]);

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
    // Listen for 'dataUpdated' events from the server
    socket.on('dataUpdated', (data) => {
      console.log('Data received:', data);

      // Filter products that match the disease received
      if (data && data.disease) {
        const filtered = products.filter(product =>
          product.targetDisease.includes(data.disease)
        ).slice(0, 5); // Limit to 5 products
        setFilteredProducts(filtered);
      }
    });

    return () => {
      socket.off('dataUpdated'); // Stop listening to the 'dataUpdated' event when the component unmounts
    };
  }, [products]);

  return (
    <div>
      <div>
        {/* <iframe 
            title="Plant Disease Detection Model" 
            src="https://plant-disease-detection-model.streamlit.app/?embed=true" 
            width="100%" 
            height="550">
        </iframe> */}
        <iframe
            src="https://dixitakhilesh-plant-disease-detection.hf.space?embeded=true"
            width="100%"
            height="600"
        ></iframe>
      </div>
      <div className='my-8'>
        {filteredProducts.length > 0 && (
          <div>
            <h1 className='font-bold text-3xl'>Recommended Products</h1>
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
        )}
      </div>
    </div>
  );
}

export default FetchResults;
