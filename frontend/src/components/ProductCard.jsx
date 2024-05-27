/* eslint-disable react/prop-types */
import { AiFillStar, AiOutlineStar, AiOutlineShopping } from "react-icons/ai";

const ProductCard = ( {img, name, price} ) => {
  return (
    <div className="border border-gray-200 hover:border-gray-300 hover:scale-105 transition-transform rounded-lg relative">
        <div className="flex justify-center items-center h-[200px] overflow-hidden">
          <img src={img} alt={name} className="max-w-[100%] max-h-[100%] object-cover" />
        </div>

        <div className="space-y-2 relative p-4">
            <div className="text-yellow-400 flex gap-[2px] text-[20px]">
                <AiFillStar />
                <AiFillStar />
                <AiFillStar />
                <AiFillStar />
                <AiOutlineStar />
            </div>

            <h3 className="font-medium text-[#333]">{name}</h3>

            <h3 className="text-2xl font-medium text-red-600">₹{price}</h3>

            <button className="absolute -top-4 right-2 bg-emerald-500 text-white text-[28px] w-[50px] h-[50px] rounded-full grid place-items-center cursor-pointer">
                <AiOutlineShopping />
            </button>
        </div>
    </div>
  )
}

export default ProductCard
