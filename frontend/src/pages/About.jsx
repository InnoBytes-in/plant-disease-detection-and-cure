import Footer from '../components/shared/Footer'
import Header from '../components/shared/Header'

const About = () => {
  return (
    <div>
      <Header />
      <div className='mx-[25%] my-[2%]'>
        <h1 className='font-bold text-3xl'>About Us</h1>
        <p className='pt-8'>Pesticides website. Check Plant Diseases. Get Weather details.</p>
      </div>
      <Footer />
    </div>
  )
}

export default About
