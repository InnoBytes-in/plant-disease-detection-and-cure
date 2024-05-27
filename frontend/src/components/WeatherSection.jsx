import { useEffect, useState } from "react";
import CurrentWeather from "../service/current-weather";
import Search from "../service/search";
import { WEATHER_API_KEY, WEATHER_API_URL } from "../service/api";

const WeatherSection = () => {
  const [currentWeather, setCurrentWeather] = useState(null);
  const [position, setPosition] = useState({ latitude: null, longitude: null });

  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(function (position) {
        setPosition({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        });
      });
    } else {
      console.log("Geolocation is not available in your browser.");
    }
  }, []);

  useEffect(() => {
    // Fetch weather data when position changes
    if (position.latitude !== null && position.longitude !== null) {
      const fetchWeather = async () => {
        try {
          const response = await fetch(
            `${WEATHER_API_URL}/weather?lat=${position.latitude}&lon=${position.longitude}&appid=${WEATHER_API_KEY}&units=metric`
          );
          if (response.ok) {
            const weatherResponse = await response.json();
            setCurrentWeather(weatherResponse);
          } else {
            console.error("Failed to fetch weather data");
          }
        } catch (error) {
          console.error("Error fetching weather data:", error);
        }
      };
      fetchWeather();
    }
  }, [position]);

  const handleOnSearchChange = (searchData) => {
    const [lat, lon] = searchData.value.split(" ");

    const currentWeatherFetch = fetch(
      `${WEATHER_API_URL}/weather?lat=${lat}&lon=${lon}&appid=${WEATHER_API_KEY}&units=metric`
    );

    Promise.all([currentWeatherFetch])
      .then(async (response) => {
        const weatherResponse = await response[0].json();

        setCurrentWeather({ city: searchData.label, ...weatherResponse });
      })
      .catch(console.log);
  };

  return (
    <div>
      <img
        className="w-full object-cover rounded-lg brightness-50"
        src="/weather.jpg"
        alt="weather image"
      />
      <div className="absolute max-w-[470px] sm:ml-12 ml-8 top-[50%] -translate-y-[50%] sm:space-y-4">
        <h2 className="text-white text-2xl sm:text-4xl md:text-6xl font-bold mb-16">
          Check Weather
        </h2>
        <div className="container">
          <div className="w-[85%] mb-8">
            <Search onSearchChange={handleOnSearchChange} />
          </div>
          {currentWeather && <CurrentWeather data={currentWeather} />}
        </div>
      </div>
    </div>
  );
};

export default WeatherSection;
