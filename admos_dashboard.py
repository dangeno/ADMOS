'''
Dashboarding ADMOS Data using Streamlit. 

Should Automatically pull from Sharepoint API


'''

#General
import pandas as pd
import glob
import numpy as np
from scipy.signal import find_peaks 
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import pywt
import peakutils
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import cumtrapz
import threading

#plotting
import matplotlib.pyplot as plt
import io
import streamlit as st
import altair as alt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable, viridis
import plotly.graph_objects as go
from plotly.io import write_image

#API Things
from msal import ConfidentialClientApplication
import webbrowser
import http.server
import socketserver
import urllib.parse
import requests
from io import BytesIO

#streamlit dashboarding
import streamlit as st




# Set up the details from your Azure app registration
client_id = '4ba70e8d-fea4-4c76-86e6-1af4eed60453'
client_secret = 'lEi8Q~Ayeq2qPj8hEKPQbhBAKoBWhoclA~N9dcr4'
tenant_id = "9798e3e4-0f1a-4f96-91ad-b31a4229413a"
authority_url = f"https://login.microsoftonline.com/{tenant_id}"
redirect_uri = 'http://localhost:55665' 
scope = ['Files.ReadWrite.All','User.Read']  # Specify the scopes your app requires


pull_token = st.sidebar.button('Pull Token')

# Check if the token already exists in the session state
if 'global_token' not in st.session_state or pull_token:
    if pull_token:
        app = ConfidentialClientApplication(
            client_id,
            authority=authority_url,
            client_credential=client_secret,
        )

        # Get the authorization request URL
        auth_url = app.get_authorization_request_url(scope, redirect_uri=redirect_uri)

        webbrowser.open(auth_url)

        class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                p = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(p.query)
                code = q.get('code')
                self.send_response(200)
                self.end_headers()
                if code:
                    self.wfile.write(b"Authentication successful! You can close this window.")
                    # Exchange the authorization code for a token
                    result = app.acquire_token_by_authorization_code(code[0], scopes=scope, redirect_uri=redirect_uri)
                    # Store the token in Streamlit's session state
                    st.session_state['global_token'] = result['access_token']
                    print('get your results here:', st.session_state['global_token'])
                    # Signal the server to stop
                    threading.Thread(target=httpd.shutdown).start()
                else:
                    self.wfile.write(b"Authentication failed or was cancelled by the user.")

        PORT = 55665
        Handler = CustomHTTPRequestHandler

        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at port {PORT}")
            httpd.serve_forever()

# Use the token from session state if it exists
if 'global_token' in st.session_state:
    # Use st.session_state['global_token'] for your operations
    st.sidebar.write('Token Found')
 
else:
    st.write('Pull Token')
    st.stop()

access_token = st.session_state['global_token']
# Headers for the request
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json'
}

# API endpoint to list items in the root folder of the OneDrive
response = requests.get('https://graph.microsoft.com/v1.0/me/drive/root/children', headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Print the items in the root directory
    items = response.json()['value']
    _='''
    for item in items:
        print(f"Name: {item['name']} - ID: {item['id']}")
    '''
else:
    print(f"Failed to list root folder: {response.status_code}")




# This function will list contents of a specified folder in your OneDrive
def list_folder_contents(folder_name, access_token):
    # Initialize the list to store file names
    file_names = []
    
    # Use the Graph API to search for the folder by name
    search_url = f"https://graph.microsoft.com/v1.0/me/drive/root/search(q='{folder_name}')"
    
    # Headers including the access token
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    # Make the search request
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        folder_contents = response.json()
        for item in folder_contents.get('value', []):
            if 'folder' in item:
                print(f"Found folder: {item['name']} with ID: {item['id']}")
                # Now list the contents of the folder
                list_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item['id']}/children"
                list_response = requests.get(list_url, headers=headers)
                if list_response.status_code == 200:
                    for file in list_response.json().get('value', []):
                        # Append the file name to the list instead of printing it
                        file_names.append(file['name'])
                break
    else:
        print("Failed to find the specified folder.")
    
    # Return the list of file names
    return file_names

files_list = list_folder_contents('ADMOS Data', access_token)


#Streamlit App interface
st.image('csi-pacific-logo.png', width = 100)
st.title('RCA ADMOS Sensor Analysis')




files = st.selectbox('select folder', files_list)
boat_class = files.split(' ')[-1]

stream_data = st.checkbox('Stream Data')

if stream_data == True:

	def stream_csv_file_content(file_name, access_token):
	    # Construct the URL to access the file's content
	    file_content_url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{file_name}:/content"
	    print(file_content_url)
	    headers = {'Authorization': f'Bearer {access_token}'}
	    
	    # Stream the file content
	    response = requests.get(file_content_url, headers=headers, stream=True)
	    
	    if response.status_code == 200:
	        return BytesIO(response.content)
	    else:
	        print(f"Failed to stream file content: {response.status_code}")
	        return None

	session_files = list_folder_contents(f'{files}', access_token)
	session_files = [file for file in session_files if file.endswith('.csv')]

	#Determine IMU and GPS file for session

	for file in session_files:
	    if 'imu' in file:
	        imu_file = file
	        process_IMU = True
	        st.write('IMU File Found')
	    elif 'gnss' in file:
	        gps_file = file
	        process_GPS = True
	        st.write('GPS File Found')


	file_stream_imu = stream_csv_file_content(f'Rowing/ADMOS Data/{files}/{imu_file}', access_token)
	file_stream_gps = stream_csv_file_content(f'Rowing/ADMOS Data/{files}/{gps_file}', access_token)

	if file_stream_imu:
	    imu_data = pd.read_csv(file_stream_imu)#, encoding='ISO-8859-1')
	else:
	    print(f"Unable to load {file_stream_imu} from the streamed content.")
	if file_stream_gps:
	    gps_data = pd.read_csv(file_stream_gps)#, encoding='ISO-8859-1')
	else:
	    print(f"Unable to load {file_stream_gps} from the streamed content.")


	def group_data(array, threshold):
	    groups = []
	    current_group = [array[0]]

	    for i in range(1, len(array)):
	        diff = array[i] - array[i - 1]
	        if diff > threshold:
	            groups.append(current_group)
	            current_group = [array[i]]
	        else:
	            current_group.append(array[i])
	    if len(current_group) >= 1000:  # Append the last group if it has at least 1000 items
	        groups.append(current_group)
	    return groups

	def align_signals(imu, gps_acc, gps_vel, lat, long):
	    """
	    Aligns the gps data with the imu data using cross-correlation and returns interpolated gps values. 
	       
	    Parameters:
	        - imu: The primary IMU signal.
	        - gps_acc: GPS acceleration data.
	        - gps_vel: GPS velocity data.
	        - lat: GPS latitude data.
	        - long: GPS longitude data.
	        
	    Returns:
	        - gps_vel_aligned: The aligned version of gps_vel.
	        - lat_aligned: The aligned version of latitude data.
	        - long_aligned: The aligned version of longitude data.
	        - offset: The offset value used for alignment.
	    """
	    
	    # Interpolate the gps data to match the length of imu
	    x_old = np.linspace(0, 1, len(gps_acc))
	    x_new = np.linspace(0, 1, len(imu))
	    gps_acc_interp = np.interp(x_new, x_old, gps_acc)
	    gps_vel_interp = np.interp(x_new, x_old, gps_vel)
	    lat_interp = np.interp(x_new, x_old, lat)
	    long_interp = np.interp(x_new, x_old, long)
	    
	    # Compute the cross-correlation between the imu and gps_acc_interp
	    cross_corr = np.correlate(imu, gps_acc_interp, mode='full')
	    lags = np.arange(-len(gps_acc_interp) + 1, len(imu))
	    
	    # Find the lag that maximizes the cross-correlation
	    offset = lags[np.argmax(cross_corr)]
	        
	    # Adjust the data arrays based on the offset
	    if offset > 0:
	        # GPS data starts before IMU
	        imu_aligned = imu[offset:].reset_index(drop=True)
	        gps_acc_aligned = gps_acc_interp[:len(imu_aligned)]
	        gps_vel_aligned = gps_vel_interp[:len(imu_aligned)]  
	        lat_aligned = lat_interp[:len(imu_aligned)]
	        long_aligned = long_interp[:len(imu_aligned)]  
	       
	    else:
	        # GPS data starts after IMU
	        offset = -offset
	        gps_acc_aligned = gps_acc_interp[offset:] 
	        gps_vel_aligned = gps_vel_interp[offset:] 
	        lat_aligned = lat_interp[offset:] 
	        long_aligned = long_interp[offset:] 
	        imu_aligned = imu[:len(gps_acc_aligned)].reset_index(drop=True)

	    
	    # Plotting for visualization
	    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)    
	    ax.plot(imu, label='IMU', color='red')
	    ax.plot(gps_acc_interp, label='GPS_ACC (Original)', color='blue', alpha=0.7)
	    ax.set_title("Original Signals")
	    ax.legend()
	    
	    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  
	    ax.plot(imu_aligned, label='IMU', color='red')
	    ax.plot(gps_acc_aligned, label='GPS_ACC_aligned', color='green', alpha=0.7)
	    ax.set_title("Aligned Signals")
	    ax.legend()
	    
	    plt.tight_layout()
	    if time_align == True:
	    	st.pyplot(plt)
	    
	    return imu_aligned, gps_vel_aligned, offset, lat_aligned, long_aligned 

	def lowpass(signal, highcut, frequency):
	    '''
	    Apply a low-pass filter using Butterworth design

	    Inputs;
	    signal = array-like input
	    high-cut = desired cutoff frequency
	    frequency = data sample rate in Hz

	    Returns;
	    filtered signal
	    '''
	    order = 2
	    nyq = 0.5 * frequency
	    highcut = highcut / nyq #normalize cutoff frequency
	    b, a = sp.signal.butter(order, [highcut], 'lowpass', analog=False)
	    y = sp.signal.filtfilt(b, a, signal, axis=0)
	    return y

	def swt_coeffs(data, wavelet, levels, coeffs_out):

	    '''
	    Perform stationary wavelet transformation on input signal

	    PyWavelet follows the “algorithm a-trous” and requires that the signal length along the transformed axis
	    be a multiple of 2**level. To ensure that the input signal meets this condition, we use numpy.pad to pad
	    the signal symmetrically, then crop the padding off of the returned coefficients.


	    Inputs;
	    data = array-like input
	    wavelet = string of specific wavelet function eg: 'bior3.1'
	    levels = int levels of decomposition
	    coeffs_out = list of desired coefficient levels eg: [0,2,9]

	    Returns:
	    dataframe containing approximation (low freq) and detail (high freq) coefficients for each specified level from coeffs_out
	    coefficients are returned with the format cAx and cDx where A is approximation, D is detail and x is level
	    '''
	    # Calculate the necessary padding size
	    target_length = np.ceil(len(data) / (2**levels)) * (2**levels)
	    padding_size = int(target_length - len(data))

	    # Pad the data symmetrically so that is a multiple of 2^levels
	    data_padded = np.pad(data, (padding_size // 2, padding_size - padding_size // 2), mode='symmetric')

	    # Perform undecimated wavelet transform
	    coeffs = pywt.swt(data_padded, wavelet=wavelet, level=levels)

	    # Create a dictionary to store the desired coefficients
	    result = {}

	    for level in coeffs_out:
	        # Get the approximation and detail coefficients for the desired level
	        approximation, detail = coeffs[level]

	        # Crop the coefficients to the original data size
	        approximation_cropped = approximation[padding_size // 2 : - (padding_size - padding_size // 2)]
	        detail_cropped = detail[padding_size // 2 : - (padding_size - padding_size // 2)]

	        # Store the coefficients in the result dictionary
	        result[f'cA{level}'] = approximation_cropped
	        result[f'cD{level}'] = detail_cropped

	    # Convert the result dictionary to a DataFrame
	    df_result = pd.DataFrame(result)

	    return df_result


	# Begin loading and processing data
	ref_speeds_stream = stream_csv_file_content(f'Rowing/WBT_reference.csv', access_token)
	ref_speeds = pd.read_csv(ref_speeds_stream)
	wbt_velocity = ref_speeds.iloc[np.where(ref_speeds['BoatClass'] == boat_class)[0][0]]['Avg_vel'] #Determine WBT velocity from boat class

	time_align = st.sidebar.checkbox('Display Time Alignment')
	

	if process_IMU and process_GPS:


	    # 6. Determine IMU and GPS sampling rate
	    time_diffs_imu = np.diff(imu_data.iloc[:, 0].values) #calculate time sample differences in microseconds
	    imu_freq_derived = int(np.ceil(np.mean(1 / (time_diffs_imu / 1000000)))) #calculate the IMU frequency (Hz) as the nearest integer rounded up
	    imu_freq = 200
	    time_diffs_gps = np.diff(gps_data.iloc[:, 1].values) #calculate time sample differences in microseconds
	    gps_freq = int(np.ceil(np.mean(1 / (time_diffs_gps / 1000000)))) #calculate the GPS frequency (Hz) as the nearest integer rounded up
	    gps_freq =10
	    scale_freq = int(imu_freq/gps_freq) # frequency scaling factor for cropping data

	    # 7. Stroke detection
	    negate = st.sidebar.checkbox('Negate Acceleration')
	    if negate == True:
	    	acc_forward = imu_data['accX [g]']*-1 #set IMU forward axis
	    elif negate == False:
	    	acc_forward = imu_data['accX [g]']*-1 #set IMU forward axis


	    #Visualize entire session in order to accurately crop data
	    res_speed = (gps_data['speedN [m/s]']**2 + gps_data['speedE [m/s]']**2)**.5

	    threshold = st.number_input('Velocity Threshold', value=2)
	    Threshold_velocity = np.where(res_speed >= threshold)[0]
	    section_indexes = group_data(Threshold_velocity, 2.0)

	    height = res_speed.max() +2

	    session_plot = go.Figure()
	    session_plot.add_trace(go.Scatter(
	        y=res_speed,
	        mode='markers',
	        marker=dict(
	            size=3,
	            color=res_speed,  # Set color to velocity
	            colorscale='Jet',  # Choose a color scale
	            colorbar=dict(title='Velocity (m/s)'),
	            opacity=0.8
	        )))

	    section_indexes = [lst for lst in section_indexes if len(lst) >= 1000]
	    section_num = 0
	    for section in section_indexes:
	        section_num += 1
	        if len(section) > 1000:
	            session_plot.add_shape(
	                type="rect",
	                x0=section[0],
	                x1=section[-1],
	                y0=0,
	                y1=height,
	                fillcolor="grey",
	                opacity=0.2)

	            session_plot.add_vline(x=section[0], line_width=3, line_dash="dash", line_color='#2ca02c',
	                                   annotation_text=f'Piece {section_num} Start',
	                                   annotation_textangle=270, annotation_position="top left", opacity=0.5)
	            session_plot.add_vline(x=section[-1], line_width=3, line_dash="dash", line_color='#d62728',
	                                   annotation_text=f'Piece {section_num} End',
	                                   annotation_textangle=270, annotation_position="top left", opacity=0.5)

	    st.plotly_chart(session_plot)

	    section_select = st.sidebar.selectbox('Select Section to Analyze', range(1, section_num+1))

	    #Select section to interpret
	    analyze = section_indexes[section_select-1]
	    analyze_append = list(np.array(analyze[:50]) - 50)
	    analyze = analyze_append + analyze
	    
	    #Crop data using interactive plotting tool (above)
	    scale_freq = 20
	    

	    start_imu = int(analyze[0]*scale_freq)
	    end_imu = int(analyze[-1]*scale_freq)
	    start_gps = int(start_imu / scale_freq)
	    end_gps = int(end_imu / scale_freq)

	    latitude = gps_data['latitude [deg]'][start_gps:end_gps].reset_index(drop=True)
	    longitude = gps_data['longitude [deg]'][start_gps:end_gps].reset_index(drop=True)

	    
	    acc_forward = acc_forward[start_imu:end_imu].reset_index(drop=True) #Crop the data to only contain one Piece
	    gps_speed = np.sqrt(gps_data['speedN [m/s]'][start_gps:end_gps]**2 +
	                            gps_data['speedE [m/s]'][start_gps:end_gps]**2) #calculate boat speed from GPS N + E vectors
	    gps_acc = np.gradient(gps_speed, edge_order=2) #Differentiate GPS vel to acc
	    

	    #Input IMU_acc, gps_acc, and gps_speed to return interpolated and aligned gps_speed
	    acc_forward, gps_speed_corrected, offset, latitude_corrected, longitude_corrected = align_signals(acc_forward, gps_acc, gps_speed, latitude, longitude)  #gps speed is now interpolated to IMU sample rate
	    
	    #Filter acc
	    acc_filt = lowpass(acc_forward,20, imu_freq) #filter acc forward axis
	    acc_max = acc_filt.max()
	    acc_min = abs(acc_filt.min())
	    acc_filt = acc_filt * (-9.81 if acc_max > acc_min else 9.81) #convert to m/s^2 and determine sign axis sign


	    # Perform undecimated (stationary) wavelet transform using swt_coeffs function
	    coeff_df = swt_coeffs(acc_filt, 'bior5.5', 5, [0, 1, 2, 3, 4]) #return the wavelet coefficients (data, wavelet, levels, coeffs_out)


	    wave_min = coeff_df['cA1'].min() #Find the minimum value of the wavelet coefficient to set threshold
	    peak_threshold = wave_min*.40 #Determine threshold value as 50% of the max peak (in this case valleys)


	    peak_method = st.sidebar.selectbox('Select Accel Peak Detection Tool', ['Wavelet Smoothed', 'Lowpass Standard', 'Wavelet Detection'])
	    if peak_method == 'Wavelet Smoothed':
	        accel_peaks, _ = find_peaks(-coeff_df['cA1'], height= -peak_threshold, distance=100) #Find peaks (valleys by negating signal and threshold)
	    if peak_method == 'Lowpass Standard':
	        accel_peaks, _ = find_peaks(-acc_filt, height=3, distance=100)
	    if peak_method == 'Wavelet Detection':
	        accel_peaks = peakutils.indexes(-coeff_df['cA1'], min_dist = 100, thres = .6)

	    # Plotting peaks with the wavelet
	    plt.figure(figsize=(10, 6), dpi=300)
	    plt.plot(coeff_df['cA1'], color='blue', label='Filtered Acceleration')
	    plt.scatter(accel_peaks, coeff_df['cA1'][accel_peaks], color='red', marker='o', label='Acceleration Peaks')
	    plt.title('Filtered Acceleration with Peaks')
	    plt.xlabel('Sample Index')
	    plt.ylabel('Acceleration (m/s^2)')
	    plt.legend()
	    plt.grid(True)
	    plt.tight_layout()
	    if time_align == True:
	    	st.pyplot(plt)


	    # 7. Process each identified stroke
	    stroke_rate = []
	    distance_per_stroke = []
	    imu_peak = pd.DataFrame()
	    gps_peak = pd.DataFrame()
	    full_strokes_df = pd.DataFrame()
	    drive_starts = []
	    drive_ends = []
	    full_stroke_lengths = []  # To store the lengths of each full_stroke
	    full_strokes_df_interp = pd.DataFrame()  # For interpolated signals
	    new_lengths = []
	    new_indexes = []

	    for i in range(len(accel_peaks) - 1):

	        initial_index = accel_peaks[i]
	        initial_length = accel_peaks[i+1] - accel_peaks[i]

	        if (800 > initial_length > 200) and (i > 0):
	            _='''
	            First we need to determine the beginning of the catch phase of the stroke
	            This becomes the initial start time for the following stroke
	            To achieve this we can reverse the array of the initial stroke, which was just the valleys associated with the drive
	            Then we can detect the first peak, as this will correspond to the end of the recovery/start of catch for the following stroke

	            '''
	            initial_stroke = coeff_df['cA0'][initial_index:initial_index + initial_length]
	            initial_stroke_reverse = pd.Series(initial_stroke.values[::-1], index=initial_stroke.index)
	            drive_start, _ = find_peaks(initial_stroke_reverse, height= 0, distance=50)
	            new_index = accel_peaks[i-1] - drive_start [0]  #this is the index for the start of the catch
	            new_length = (initial_index - drive_start[0]) - new_index
	            new_lengths.append(new_length)  # Append the new_length to the list
	            new_indexes.append(new_index)  # Append the new_index to the list

	            full_stroke = acc_filt[new_index:new_index+new_length]
	            full_stroke_series = pd.Series(full_stroke, name=f'Stroke {i}').reset_index(drop=True)


	            if (800 > new_length > 200) and (i > 0):
	                full_strokes_df = pd.concat([full_strokes_df, full_stroke_series], axis=1)
	                full_stroke_lengths.append(len(full_stroke))
	                drive_starts.append(drive_start[0])


	def interpolate_to_max_length(data, max_length, method='linear'):
	    """
	    Interpolates each column of the dataframe to the specified max_length using the specified method.
	    :param data: DataFrame containing the data to be interpolated.
	    :param max_length: The target length for the interpolation.
	    :param method: The interpolation method ('linear', 'nearest', 'zero', 'slineasr', 'quadratic', 'cubic', etc.)
	    :return: DataFrame containing the interpolated data.
	    """
	    interpolated_data_list = []

	    for column in data.columns:
	        # Drop NaN values
	        data_cleaned = data[column].dropna()
	        # Original length of the cleaned data
	        original_length = len(data_cleaned)
	        # Original indices normalized between 0 and 1
	        original_indices = np.linspace(0, 1, original_length)
	        # New indices based on the max_length, also normalized between 0 and 1
	        new_indices = np.linspace(0, 1, max_length)

	        # Interpolation using scipy
	        f = interp.interp1d(original_indices, data_cleaned, kind=method, fill_value="extrapolate")
	        interpolated_column = f(new_indices)

	        interpolated_data_list.append(pd.Series(interpolated_column, name=column))

	    interpolated_df = pd.concat(interpolated_data_list, axis=1)

	    return interpolated_df

	def stroke_features(data, wavelet, levels, coeffs_out, new_lengths):
	    '''
	    Input data should be a dataframe containing individual strokes
	    The default wavelet choice should be 'bior4.4'
	    The default levels should be 9
	    The default coeffs_out should be [3]
	    This function uses wavelet decomposition + peak detection to identify the end of drive phase
	    Returns a list of indices for drive end for each stroke
	    Must be called separately for time normalized strokes and raw stroke data
	    '''
	    drop_drive = []

	    plt.figure(figsize=(10, 6))
	
	    for i, column in enumerate(data.columns):
	        stroke = data[column].dropna()
	        stroke_wave1 = swt_coeffs(stroke, wavelet, levels, coeffs_out)
	        stroke_wave = stroke_wave1['cA2'].iloc[::-1]
	        stroke_wave_der = lowpass(np.diff(stroke) / (1/imu_freq), 2,10)
	        ax = sns.scatterplot( x = range(0, len(stroke_wave)), y= list(stroke_wave))

	        #stroke_peak, _ = find_peaks(-stroke_wave_der, height=-20, distance=10)
	        #stroke_peak, _ = find_peaks(-stroke_wave, height= 0, distance=10)
	        stroke_peak = peakutils.indexes(-stroke_wave, thres= 0.2, min_dist=10)
	        # Check if any peaks were detected



	        if len(stroke_peak) > 0:
	            #drive_end = len(stroke_wave) - stroke_peak[-1]
	            drive_end = stroke_peak[0] + (len(stroke_wave) - stroke_peak[0])
	            drive_end = (len(stroke_wave) - stroke_peak[0])
	            #drive_end = stroke_peak[-1]
	            drive_ends.append(drive_end)
	        else:
	            # If no peaks were detected, append None or any other suitable value
	            drive_ends.append(None)
	            drop_drive.append(i)
	            print("No Peak Detected")

	        plt.axvline(x = stroke_peak[0], color = 'b')


	        #drive_end = new_lengths[i] - stroke_peak[0]
	        #drive_ends.append(drive_end)
	        
	    #st.pyplot(plt)

	    return drive_ends, drop_drive

	def cropped_strokes_df(velocity, indices, lengths, sample_rate, latitude, longitude,):
	    """
	    Analyzes GPS data for each stroke.

	    Args:
	    - data: numpy array containing the GPS data.
	    - indices: List of starting indices for each stroke.
	    - lengths: List of lengths for each stroke.
	    - sample_rate: The sampling rate of the data.

	    Returns:
	    - cropped_strokes_df: DataFrame containing cropped data for each stroke.
	    - avg_velocities: Series containing the average velocity for each stroke.
	    - distances: Series containing the distance covered in each stroke.
	    """

	    cropped_strokes = []
	    avg_velocities_list = []
	    distances_list = []
	    distances = []

	    for idx, length in zip(indices, lengths):
	        cropped_data_array = velocity[idx:idx+length]
	        cropped_data_series = pd.Series(cropped_data_array).reset_index(drop=True)

	        # Crop the latitude and longitude for the current stroke
	        stroke_lat = latitude[idx:idx+length]
	        stroke_long = longitude[idx:idx+length]

	        avg_velocity = cropped_data_series.mean()

	        # Calculate the distance using the cumulative trapezoidal rule and get the last value for total distance
	        distance = cumulative_trapezoid(cropped_data_series, dx=1/sample_rate)[-1]

	        cropped_strokes.append(cropped_data_series)
	        avg_velocities_list.append(avg_velocity)
	        distances_list.append(distance)

	        # Calculate the distance using lat + long
	        lat_diff = (stroke_lat[-1] - stroke_lat[0]) * 111139
	        long_diff = (stroke_long[-1] - stroke_long[0]) * 111139
	        dist = np.sqrt(lat_diff**2 + long_diff**2)

	        distances.append(dist)

	    cropped_strokes_df = pd.concat(cropped_strokes, axis=1)
	    avg_velocities = pd.Series(avg_velocities_list)
	    distance_list = pd.Series(distances_list)

	    return cropped_strokes_df, avg_velocities, distance_list, distances

	def gps_stroke_analysis(velocity, indices, lengths, sample_rate, latitude, longitude,):
	    """
	    Analyzes GPS data for each stroke.

	    Args:
	    - data: numpy array containing the GPS data.
	    - indices: List of starting indices for each stroke.
	    - lengths: List of lengths for each stroke.
	    - sample_rate: The sampling rate of the data.

	    Returns:
	    - cropped_strokes_df: DataFrame containing cropped data for each stroke.
	    - avg_velocities: Series containing the average velocity for each stroke.
	    - distances: Series containing the distance covered in each stroke.
	    """

	    cropped_strokes = []
	    avg_velocities_list = []
	    distances_list = []
	    distances = []


	    for idx, length in zip(indices, lengths):
	        cropped_data_array = velocity[idx:idx+length]
	        cropped_data_series = pd.Series(cropped_data_array).reset_index(drop=True)

	        # Crop the latitude and longitude for the current stroke
	        stroke_lat = latitude[idx:idx+length]
	        stroke_long = longitude[idx:idx+length]

	        avg_velocity = cropped_data_series.mean()

	        # Calculate the distance using the cumulative trapezoidal rule and get the last value for total distance
	        distance = cumulative_trapezoid(cropped_data_series, dx=1/sample_rate)[-1]

	        cropped_strokes.append(cropped_data_series)
	        avg_velocities_list.append(avg_velocity)
	        distances_list.append(distance)

	        # Calculate the distance using lat + long
	        lat_diff = (stroke_lat[-1] - stroke_lat[0]) * 111139
	        long_diff = (stroke_long[-1] - stroke_long[0]) * 111139
	        dist = np.sqrt(lat_diff**2 + long_diff**2)

	        distances.append(dist)

	    cropped_strokes_df = pd.concat(cropped_strokes, axis=1)
	    avg_velocities = pd.Series(avg_velocities_list)
	    distance_list = pd.Series(distances_list)



	    return cropped_strokes_df, avg_velocities, distance_list, distances



	#Identify stroke features for drive and catch for original data
	#drive_ends_raw, drop_drive_raw = stroke_features(full_strokes_df, 'bior4.4', 9, [0,1,2,3], new_lengths)

	#drive_ends_raw = pd.Series(drive_ends_raw).drop(index = drop_drive_raw)
	#drive_starts_raw = pd.Series(drive_starts).drop(index = drop_drive_raw)


	#drive_time = (np.array(drive_ends_raw)-np.array(drive_starts_raw))/imu_freq



	#generate interactive plot that breaks down stroke velocities


	#Time normalize strokes
	max_length = 1000 #int(sum(full_stroke_lengths) / len(full_stroke_lengths)) #Interpolate to average stroke length if desired

	full_strokes_df_interp = interpolate_to_max_length(full_strokes_df, max_length)

	#Identify stroke features for drive and catch for interpolated data
	new_lengths_interp = [max_length] * len(full_strokes_df_interp.columns) #Input list for the length of each stroke used in stroke_features function

	_='''
	drive_start_interp =( drive_start_original/stroke_length_original)*length_interpolated
	where:
	drive_start_original = drive_start
	stroke_length_original = new_length
	length_interpolated = max_length

	'''
	drive_start_interp = []
	for start, origlen, interplen in zip(drive_starts, new_lengths, new_lengths_interp):
	    proportion = start / origlen
	    adjusted_drive_start = proportion * interplen
	    drive_start_interp.append(int(adjusted_drive_start))

	drive_end_interp,drop_drive_interp = stroke_features(full_strokes_df_interp, 'bior4.4', 9, [0,1,2,3], new_lengths_interp)

	percent_DE = drive_end_interp
	percent_DS = drive_start_interp

	# Compute the average stroke waveform
	average_stroke_waveform_interp = full_strokes_df_interp.mean(axis=1)

	# Calculate average drive start and drive end
	avg_drive_start = np.mean(pd.Series(drive_start_interp).drop(index = drop_drive_interp))
	avg_drive_end = np.mean(pd.Series(drive_end_interp).drop(index = drop_drive_interp))


	#Calculate GPS stroke velocity and distance
	gps_strokes, gps_velocities, integrated_distance, gps_distance = gps_stroke_analysis(gps_speed_corrected,
	                                                                new_indexes, new_lengths, imu_freq, latitude_corrected, longitude_corrected)

	#Interpolate GPS strokes
	gps_strokes_interp = interpolate_to_max_length(gps_strokes, max_length)

	#Calculate average GMS scaled stroke velocity
	gms = gps_velocities / wbt_velocity

	#Calculate average stroke rate
	#stroke_rate = 60 / (pd.Series(new_lengths).mean() / imu_freq)
	stroke_rate = 60 / (pd.Series(new_lengths) / imu_freq)
	#Calculate distance
	total_distance = sum(gps_distance)
	average_vel = gps_velocities.mean()

	#Calculate GMS scaled strokes
	stroke_velocity_gms = gps_strokes / wbt_velocity



	stroke_number = 0
	stroke_min_vel = []
	stroke_max_vel = []
	stroke_mean_vel = []
	stroke_mean_accel = []

	#plot Stoke Overlay
	stroke_plot = go.Figure()

	for stroke in gps_strokes.columns:
	    stroke_number += 1

	    stroke_min_vel.append(gps_strokes[stroke].min())
	    stroke_max_vel.append(gps_strokes[stroke].max())


	    stroke_plot.add_trace(go.Scatter(
	        y=[stroke_number] * len(gps_strokes[stroke]),
	        x=np.array(range(0, len(gps_strokes[stroke]))),
	        mode='markers',
	        marker=dict(
	            size=3,
	            color=gps_strokes[stroke]/ wbt_velocity,  # Set color to velocity
	            colorscale='Jet', 
	            cmin=0.25,  # Fixed minimum value of your color scale
        		cmax=1.25, # Choose a color scale
	            coloraxis="coloraxis",  # Reference to a shared color axis
	            opacity=0.8
	        )
	    ))

	# Update layout to include a shared coloraxis (colorbar) settings
	stroke_plot.update_layout(
	    title="Stacked Stroke Visualization",
	    xaxis_title="Length of Stroke",
	    yaxis_title="Stroke Number",
	    showlegend=False,  # This line removes the legend
	    yaxis=dict(tickmode='array'),
	    coloraxis=dict(
	        colorbar=dict(title="Velocity (m/s)"),
	        colorscale='Jet'
	    )
	)
	st.plotly_chart(stroke_plot)
	#If memory becomes an issue
	_='''
	gps_data = gps_strokes.unstack().reset_index()
	gps_data.columns = ['stroke_number', 'position_in_stroke', 'velocity']
	gps_data = gps_data.dropna()
	#st.write(gps_data)
	# Create the plot
	plt.figure(figsize=(10, 6))
	ax = sns.scatterplot(data=gps_data, x='position_in_stroke', y='stroke_number', hue='velocity', palette='viridis', s=50, edgecolor='none', legend=False)

	# Normalize the colormap
	norm = Normalize(vmin=gps_data['velocity'].min(), vmax=gps_data['velocity'].max())
	sm = ScalarMappable(norm=norm, cmap='turbo')

	# Add colorbar
	plt.colorbar(sm, ax=ax, label='Velocity (m/s)')

	# Setting the plot title and labels
	ax.set_title('Stacked Stroke Visualization')
	ax.set_xlabel('Position in Stroke')
	ax.set_ylabel('Stroke Number')
	#st.pyplot(plt)
	'''
	#Calculate average stroke velocity waveform
	average_stroke_gps_interp = gps_strokes_interp.mean(axis=1)

	#Calculate average GMS scaled stroke velocity waveform
	average_stroke_gms_interp = average_stroke_gps_interp / wbt_velocity


	def plot_average_waveform(data_sel, waveform, acc_waveform, avg_drive_start, avg_drive_end, title, y_label):
	    """
	    Plots the average waveform with annotations for drive start and drive end.

	    Parameters:
	        - waveform: The average waveform to be plotted (pandas Series).
	        - avg_drive_start: Index of the average drive start.
	        - avg_drive_end: Index of the average drive end.
	        - title: Title for the plot.
	        - y_label: Label for the y-axis.

	    Returns:
	        None
	    
	    plt.figure(figsize=(10, 6), dpi=300)
	    plt.plot(waveform, label='Average Waveform', color='blue')
	    plt.axvline(x=avg_drive_start, color='green', linestyle='--', label='Avg Drive Start')
	    plt.axvline(x=avg_drive_end, color='red', linestyle='--', label='Avg Drive End')
	    plt.scatter(avg_drive_start, waveform.iloc[int(avg_drive_start)], color='red', s=100, zorder=5)
	    plt.scatter(avg_drive_end, waveform.iloc[int(avg_drive_end)], color='green', s=100, zorder=5)
	    plt.title(title)
	    plt.xlabel('Sample Index')
	    plt.ylabel(y_label)
	    plt.legend()
	    plt.tight_layout()
	    st.pyplot(plt)
	    """
	    if data_sel == 'Acceleration':
	        waveform_df = pd.DataFrame(acc_waveform)
	        waveform_df.columns = ['Waveform']
	        waveform_df['Sample Index'] = waveform_df.index
	        y_axis = 'Boat Acceleration'
	    elif data_sel == 'Velocity': 
	        waveform_df = pd.DataFrame(waveform)
	        waveform_df.columns = ['Waveform']
	        waveform_df['Sample Index'] = waveform_df.index
	        y_axis = 'Boat Velocity (GMS)'


	    min_waveform = waveform_df['Waveform'].min()
	    max_waveform = waveform_df['Waveform'].max()
	    buffer = (max_waveform - min_waveform) * 0.05 

	    # Creating points for avg_drive_start and avg_drive_end on the waveform
	    points_df = pd.DataFrame({
	    'Sample Index': [avg_drive_start, avg_drive_end],
	    'Waveform': [waveform_df.loc[int(avg_drive_start), 'Waveform'], waveform_df.loc[int(avg_drive_end), 'Waveform']],
	    'Color': ['red', 'green']  # This will help in differentiating the points
	    })

	    # Base chart for the waveform
	    waveform_chart = alt.Chart(waveform_df).mark_line().encode(
	    x='Sample Index',
	    y=alt.Y('Waveform', scale=alt.Scale(domain=[min_waveform - buffer, max_waveform + buffer]), title=y_axis),
	    color=alt.value('blue'))


	    # Adding vertical lines for avg_drive_start and avg_drive_end
	    vline_chart = alt.Chart(pd.DataFrame({
	    'Sample Index': [avg_drive_start, avg_drive_end],
	    'Color': ['green', 'red']  # Different colors for start and end
	    })).mark_rule(strokeDash=[5,5]).encode(
	    x='Sample Index',
	    color=alt.Color('Color', scale=None)  # Using the color field directly
	    )

	    # Adding scatter points for avg_drive_start and avg_drive_end on the waveform
	    points_chart = alt.Chart(points_df).mark_point(filled=True, size=100).encode(
	    x='Sample Index',
	    y='Waveform',
	    color=alt.Color('Color', scale=None)  # Using the color field directly
	    )

	    # Combine the charts
	    final_chart = waveform_chart  + vline_chart + points_chart


	    # Add titles and labels
	    final_chart = final_chart.properties(
	    width=600,  # Altair works with pixel sizes directly
	    height=360,
	    title=title
	    ).configure_axis(
	    labelFontSize=12,
	    titleFontSize=14
	    )



	    st.altair_chart(final_chart)

	    return final_chart



	# Plot the average stroke waveforms
	#plot_average_waveform(average_stroke_waveform_interp, avg_drive_start, avg_drive_end, 'Average Stroke Waveform Acceleration', 'Acceleration (m/s^2)')

	#plot_average_waveform(average_stroke_gps_interp, avg_drive_start, avg_drive_end, 'Average GPS Stroke Velocity', 'Velocity (m/s)')


	data_sel = st.sidebar.selectbox('Select Data Visualization', ['Acceleration', 'Velocity'])
	final_chart = plot_average_waveform(data_sel, average_stroke_gms_interp, average_stroke_waveform_interp, avg_drive_start, avg_drive_end, 'Average GMS Scaled Stroke Velocity', '% GMS Velocity')


	df_exports = pd.DataFrame()
	df_exports['percent_drive_start'] = drive_start_interp
	df_exports['percent_drive_end'] = drive_end_interp
	df_exports['drive_time'] = (((df_exports['percent_drive_end']-df_exports['percent_drive_start']))/1000)*pd.Series(new_lengths)/imu_freq
	df_exports['stroke_rate'] = stroke_rate
	df_exports['distance_per_stroke'] = integrated_distance
	df_exports['stroke_max_vel'] = stroke_max_vel
	df_exports['stroke_min_vel'] = stroke_min_vel
	def convert_time(seconds):
	    # Calculate the minutes, seconds, and milliseconds
	    minutes = int(seconds // 60)
	    seconds_remainder = int(seconds % 60)
	    milliseconds = int((seconds - int(seconds)) * 1000)
	    
	    # Format the string with leading zeros if necessary
	    return f"{minutes:02d}:{seconds_remainder:02d}:{milliseconds:03d}"

	tab1, tab2, tab3 = st.tabs(["Stroke Rate", "Distance Per Stroke", "Velocity and Time"])
	with tab1:   
	    st.subheader('Stroke Rate')
	    col1, col2, col3, col4 = st.columns(4)
	    with st.container():
	        with col1: 
	            st.metric('Average Stroke Rate', round(stroke_rate.mean(), 2))
	        with col2: 
	            st.metric('Max Stroke Rate', round(stroke_rate.max(), 2))
	        with col3: 
	            st.metric('Min Stroke Rate', round(stroke_rate.min(), 2))
	        with col4: 
	            st.metric('STD Stroke Rate', round(stroke_rate.std(), 2))

	with tab2:    
	    st.subheader('Distance Per Stroke')
	    col5, col6, col7, col8 = st.columns(4)
	    with st.container():
	        with col5: 
	            st.metric('Average DPS', round(df_exports['distance_per_stroke'].mean(), 2))
	        with col6: 
	            st.metric('Max DPS', round(df_exports['distance_per_stroke'].max(), 2))
	        with col7: 
	            st.metric('Min DPS', round(df_exports['distance_per_stroke'].min(), 2))
	        with col8: 
	            st.metric('STD DPS', round(df_exports['distance_per_stroke'].std(), 2))


	with tab3: 
	    st.subheader('Velocity and Timing Summaries')

	    col9, col10, col11, col12 = st.columns(4)
	    with st.container():
	        with col9: 
	            st.metric('Average Max Velocity', round(df_exports['stroke_max_vel'].mean(), 2))
	            st.metric('Average Pace / 500m', convert_time((accel_peaks[-1]- accel_peaks[0])/imu_freq/np.sum(df_exports['distance_per_stroke'])*500))
	        with col10: 
	            st.metric('Average Min Velocity', round(df_exports['stroke_min_vel'].mean(), 2))
	            st.metric('Average Boat Velocity', round(average_vel, 2))
	            
	            
	        with col11: 
	            st.metric('Session Time', convert_time(((imu_data['imuTimestamp [us]'][accel_peaks[-1]]- imu_data['imuTimestamp [us]'][accel_peaks[0]])/1000000)))
	            st.metric('Average Drive Time', round(df_exports['drive_time'].mean(), 2))

	        with col12: 

	        	st.metric('Total Distance', round(average_vel*((imu_data['imuTimestamp [us]'][accel_peaks[-1]]- imu_data['imuTimestamp [us]'][accel_peaks[0]])/1000000)))

	#st.write(df_exports)

else: 
	st.header('Select Data for Analysis')

