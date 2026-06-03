CREATE TABLE areas_of_interest (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    layer_type VARCHAR(30) DEFAULT 'street_block',
    coordinates JSONB NOT NULL -- Holds standard GeoJSON array blocks
);

-- 2. Infrastructure Point Sources (Factories / Power Plants)
CREATE TABLE pollution_sensors (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    layer_type VARCHAR(30) DEFAULT 'emission_source',
    emblem_type VARCHAR(30) NOT NULL, -- e.g., 'factory', 'power_plant'
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL
);