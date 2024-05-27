import Footer from "../components/shared/Footer"
import Header from "../components/shared/Header"

const Contact = () => {
  return (
    <div>
      <Header />
      <div className='mx-[25%] my-[2%]'>
        <h1 className='font-bold text-3xl'>Contact Us</h1>
        <p className='pt-8'>Phone No.: 9876543210</p>
        <p className='pt-4'>Address: Kondhwa, Pune</p>
      </div>
      <Footer />
    </div>
  )
}

export default Contact
