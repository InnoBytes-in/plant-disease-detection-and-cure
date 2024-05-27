import ProductCard from "./ProductCard";

const data = [
    { id: 0, img: "product_1.jpg", name: "Product 1", price: "50" },
    { id: 1, img: "product_2.png", name: "Product 2", price: "30" },
    { id: 2, img: "product_3.jpg", name: "Product 3", price: "100" },
    { id: 3, img: "product_4.jpg", name: "Product 4", price: "5" },
    { id: 4, img: "product_5.png", name: "Product 5", price: "20" },
];

const FeatureSection = () => {
  return (
    <div className="container p-16">
        <div className="justify-between items-center">
            <h3 className="font-bold text-2xl">Top Products</h3>

            <div className="grid sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-5 pt-8 gap-5">
                {data.map((el) => 
                    <ProductCard
                        key={el.id}
                        img={el.img}
                        name={el.name}
                        price={el.price} 
                    />
                )}
            </div>
        </div>
    </div>
  )
}

export default FeatureSection
