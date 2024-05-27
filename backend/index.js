import express from "express";
import dotenv from "dotenv";
import colors from 'colors';
import cors from 'cors';
import connectDB from "./config/db.js";

import productRoutes from './routes/productRoutes.js';

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;
connectDB();

app.use(express.json());
app.use(cors());

app.use('/products', productRoutes);

app.listen(PORT, console.log(`Server running on port ${PORT}`.yellow.bold));
