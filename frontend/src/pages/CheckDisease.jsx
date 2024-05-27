import FetchResults from "../components/FetchResults"
import Header from "../components/shared/Header"

const CheckDisease = () => {
  return (
    <div>
      <Header />
      <div className="container mt-8">
        <div className="justify-between items-center mx-16">
            <FetchResults />
        </div>
      </div>
    </div>
  )
}

export default CheckDisease
