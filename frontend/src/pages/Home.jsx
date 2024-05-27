import BlogSection from "../components/BlogSection"
import FeatureSection from "../components/FeatureSection"
import Hero from "../components/Hero"
import Header from "../components/shared/Header"

const Home = () => {
  return (
    <div>
      <Header />
      <Hero />
      <FeatureSection />
      <BlogSection />
    </div>
  )
}

export default Home
