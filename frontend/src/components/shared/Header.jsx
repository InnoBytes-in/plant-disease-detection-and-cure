import { useState } from 'react';
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react';
import { Link, NavLink } from 'react-router-dom';
import { FaCartShopping } from 'react-icons/fa6';
import { FaBars, FaTimes } from 'react-icons/fa';

const Header = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <nav>
      <div className='bg-whitesmoke duration-200 relative z-40 py-4'>
        <div className='container flex justify-between items-center'>
          <div className='mx-6'>
            <Link to="/"><img className='h-6' src="/logo.png" alt='Green Shield logo' /></Link>
          </div>
          <div className='hidden lg:flex list-none'>
            <ul className="flex gap-20">
              <li>
                <NavLink className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/">Home</NavLink>
              </li>
              <li>
                <NavLink className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/about">About</NavLink>
              </li>
              <li>
                <NavLink className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/check-disease">Check Disease™</NavLink>
              </li>
              <li>
                <NavLink className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/products">Products</NavLink>
              </li>
              <li>
                <NavLink className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/contact">Contact</NavLink>
              </li>
            </ul>
          </div>
          <div className='flex items-center mx-6 gap-6'>
            <div>
              <button className="relative icon_wrapper">
                <FaCartShopping className="text-xl text-gray-600" />
                <div className="w-4 h-4 bg-red-500 text-white rounded-full absolute top-0 right-0 flex items-center justify-center text-xs">
                  4
                </div>
              </button>
            </div>
            <div>
              <SignedIn>
                <UserButton afterSignOutUrl="/" />
              </SignedIn>
              <SignedOut>
                <Link className='btn' to="/sign-in">Sign In</Link>
              </SignedOut>
            </div>
            <div className='lg:hidden'>
              <button onClick={toggleMobileMenu} className='text-xl text-gray-600'>
                {isMobileMenuOpen ? <FaTimes /> : <FaBars />}
              </button>
            </div>
          </div>
        </div>
        {isMobileMenuOpen && (
          <div className='lg:hidden list-none bg-whitesmoke p-4'>
            <ul className="flex flex-col gap-4">
              <li>
                <NavLink onClick={toggleMobileMenu} className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/">Home</NavLink>
              </li>
              <li>
                <NavLink onClick={toggleMobileMenu} className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/about">About</NavLink>
              </li>
              <li>
                <NavLink onClick={toggleMobileMenu} className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/check-disease">Check Disease™</NavLink>
              </li>
              <li>
                <NavLink onClick={toggleMobileMenu} className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/products">Products</NavLink>
              </li>
              <li>
                <NavLink onClick={toggleMobileMenu} className={(e) => {return e.isActive? "active-nav-link": "nav-link"}} to="/contact">Contact</NavLink>
              </li>
            </ul>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Header;
