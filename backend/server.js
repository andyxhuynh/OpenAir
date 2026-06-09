const express = require('express');
const cors = require('cors'); // Crucial for allowing frontend access
const app = express();
const PORT = 5000;

// Enable CORS so your frontend file can talk to this server
app.use(cors());

// Define the API endpoint
app.get('/api/pollution', (req, res) => {
    const liveData = [
        { name: "Factory Alpha", lat: 45.7830, lng: 4.8820, pollution: "high" }
    ];
    // Send data back to the frontend as JSON
    res.json(liveData);
});

app.listen(PORT, () => {
    console.log(`Backend server running on http://localhost:${PORT}`);
});