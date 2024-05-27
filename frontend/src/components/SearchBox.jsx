import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const SearchBox = () => {
  const navigate = useNavigate();
  const [keyword, setKeyword] = useState('');

  const submitHandler = (e) => {
    e.preventDefault();
    if (keyword.trim()) {
      navigate(`/search/${keyword}`);
    } else {
      navigate('/');
    }
  };

  return (
    <form onSubmit={submitHandler} className='flex'>
        <input type='text' name='q' onChange={(e) => setKeyword(e.target.value)} placeholder='Search Products...' className='mx-5' />
        <button type='submit' className='p-2'>Search</button>
    </form>
  );
};

export default SearchBox;
