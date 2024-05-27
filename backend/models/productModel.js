import mongoose from 'mongoose';

const productSchema = mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    image: String,
    chemicalCompound: String,
    manufacturer: String,
    targetDisease: [String],
    targetPlant: [String],
    price: {
        type: Number,
        required: true,
        default: 0,
    },
    countInStock: {
        type: Number,
        default: 0,
    },
    rating: {
        type: Number,
        default: 0,
    },
    createdAt: {
        type: Date,
        default: new Date()
    },
    updatedAt: {
        type: Date,
        default: new Date()
    }
});

const Product = mongoose.model('Product', productSchema);

export default Product;