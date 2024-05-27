import axios from 'axios';

const url = import.meta.env.VITE_BACKEND_DEPLOYMENT_URL + "/products";

export const fetch = () => axios.get(url).then((res) => {
    return res;
});

export const createProduct = (newProduct) => axios.post(url, newProduct);