import { BsArrowRight } from "react-icons/bs";
import { Link } from "react-router-dom";
import WeatherSection from "./WeatherSection";

const Hero = () => {
  return (
    <div className="container">
      <div className="xl:flex xl:flex-row mx-[7%] gap-[5%]">
        <div className="xl:basis-2/3 relative">
            <img 
                className="w-full object-cover rounded-lg brightness-50"
                src="/check_disease.jpeg"
                alt="check disease image"
            />
            <div className="absolute max-w-[470px] sm:ml-16 ml-8 top-[50%] -translate-y-[50%] sm:space-y-4">
                <h2 className="text-white text-2xl sm:text-4xl md:text-6xl font-bold">Check Plant Disease</h2>
                <p className="text-white text-xl pt-4 sm:pt-8">Upload plant disease image and get to know the disease and its solution</p>
                <div>
                    <Link to="/check-disease">
                        <div className="bg-green-900 hover:bg-green-600 text-white rounded-full w-fit flex items-center px-4 py-2 sm:px-6 sm:py-3 gap-4 text-[14px] cursor-pointer">
                            Check Now <BsArrowRight />
                        </div>
                    </Link>
                </div>
            </div>
        </div>

        <div className="relative xl:basis-1/3">
            <WeatherSection />
        </div>
      </div>
    </div>
  )
}

export default Hero