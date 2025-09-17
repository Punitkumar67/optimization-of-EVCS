from flask import Flask, render_template, jsonify
from datetime import datetime
import os

app = Flask(__name__)

# Global status for testing
test_status = {
    'status': 'idle',
    'progress': 0,
    'current_step': 0,
    'logs': [],
    'goa': None,
    'distance': None,
    'evcs': None
}

# remove old data files if they exist
data_files = [
    './data/final_maxdispersion_10_evcs.csv',
    './data/DG_SC_placement_locationwise.csv',
    './data/gurugram_50loc_distance_matrix.csv'
    ]
for file in data_files:
    if os.path.exists(file):
        os.remove(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test')
def test_connection():
    return jsonify({
        'status': 'success',
        'message': 'Backend connection working!',
        'timestamp': str(datetime.now())
    })

@app.route('/api/locations')
def get_locations():
    # get locations from the final file final_maxdispersion_10_evcs.csv
    try:
        with open('./data/final_maxdispersion_10_evcs.csv', 'r') as f:
            # Type,BusNo,Location,Size_kW_or_kVAR,AvgEVPerDay,AvgSOC_Arrival,AvgSOC_Departure,AvgBattery_kWh,NumChargingPoints,LandCost_x,ProximityToHighway_x,PopulationDensity_x,ElectricityCost,ChargingEfficiency,AnnualElecCost,AnnualElecCost_norm,LandCost_norm,LandCost_y,PopulationDensity_y,NearbyCommercial,ProximityToHighway_y,NearbyEVSeller,EVDensity,MallsNearby,NearbyEVSeller_norm,EVDensity_norm,MallsNearby_norm,WeightedSum,Latitude,Longitude
            locations = []
            for line in f.readlines()[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) < 12:
                    continue
                location = {
                    # Type,BusNo,Location,Size_kW_or_kVAR,AvgEVPerDay,AvgSOC_Arrival,AvgSOC_Departure,AvgBattery_kWh,NumChargingPoints,LandCost_x,ProximityToHighway_x,PopulationDensity_x,ElectricityCost,ChargingEfficiency,AnnualElecCost,AnnualElecCost_norm,LandCost_norm,LandCost_y,PopulationDensity_y,NearbyCommercial,ProximityToHighway_y,NearbyEVSeller,EVDensity,MallsNearby,NearbyEVSeller_norm,EVDensity_norm,MallsNearby_norm,WeightedSum,Latitude,Longitude include everything except Type
                    'BusNo': parts[1],
                    'Location': parts[2],
                    'Size_kW_or_kVAR': parts[3],
                    'AvgEVPerDay': parts[4],
                    'AvgSOC_Arrival': parts[5],
                    'AvgSOC_Departure': parts[6],
                    'AvgBattery_kWh': parts[7],
                    'NumChargingPoints': parts[8],
                    'LandCost_x': parts[9],
                    'ProximityToHighway_x': parts[10],
                    'PopulationDensity_x': parts[11],
                    'ElectricityCost': parts[12],
                    'ChargingEfficiency': parts[13],
                    'AnnualElecCost': parts[14],
                    'AnnualElecCost_norm': parts[15],
                    'LandCost_norm': parts[16],
                    'LandCost_y': parts[17],
                    'PopulationDensity_y': parts[18],
                    'NearbyCommercial': parts[19],
                    'ProximityToHighway_y': parts[20],
                    'NearbyEVSeller': parts[21],
                    'EVDensity': parts[22],
                    'MallsNearby': parts[23],
                    'NearbyEVSeller_norm': parts[24],
                    'EVDensity_norm': parts[25],
                    'MallsNearby_norm': parts[26],
                    'WeightedSum': parts[27],
                    'lat': parts[28],
                    'lng': parts[29]
                }
                locations.append(location)
            return jsonify(locations)
    except FileNotFoundError:
        # return jsonify(SAMPLE_LOCATIONS)
        return jsonify({'error': 'Data file not found'}), 404

@app.route('/api/run_optimization', methods=['POST'])
def run_optimization():
    global test_status
    
    if test_status['status'] == 'running':
        return jsonify({'error': 'Optimization already running'}), 400
    
    # Start optimization
    def optimization():
        global test_status
        test_status['status'] = 'running'
        test_status['progress'] = 0
        test_status['current_step'] = 1
        test_status['logs'] = ['Starting optimization...']
        
        import time
        
        # Simulate GOA step
        # run goa.py 
        import GOA
        for i in range(0, 40, 10):
            test_status['progress'] = i
            test_status['logs'].append(f'GOA progress: {i}%')
            time.sleep(0.2)
        
        # Simulate Distance step
        import virtual_distance
        test_status['current_step'] = 2
        test_status['logs'].append('Starting distance matrix generation...')
        for i in range(40, 70, 10):
            test_status['progress'] = i
            test_status['logs'].append(f'Distance progress: {i}%')
            time.sleep(0.2)
        
        # Simulate EVCS step
        import best_evcs
        test_status['current_step'] = 3
        test_status['logs'].append('Starting EVCS optimization...')
        for i in range(70, 100, 10):
            test_status['progress'] = i
            test_status['logs'].append(f'EVCS progress: {i}%')
            time.sleep(0.2)
        
        # Complete
        test_status['status'] = 'completed'
        test_status['progress'] = 100
        test_status['current_step'] = 4
        test_status['logs'].append('Optimization completed!')
        test_status['goa'] = {'stdout': 'GOA completed successfully', 'stderr': '', 'returncode': 0}
        test_status['distance'] = {'stdout': 'Distance matrix generated', 'stderr': '', 'returncode': 0}
        test_status['evcs'] = {'stdout': 'EVCS optimization completed', 'stderr': '', 'returncode': 0}
    
    # Run in background thread
    import threading
    thread = threading.Thread(target=optimization)
    thread.start()
    
    return jsonify({'message': 'Optimization started'})

@app.route('/api/status')
def get_status():
    return jsonify(test_status)

if __name__ == '__main__':
    print("🚀 Starting Power Grid Optimization Dashboard (Test Version)")
    print("📍 Access at: http://localhost:5000")
    print("🔧 This is a test version with sample data")
    print("-" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)