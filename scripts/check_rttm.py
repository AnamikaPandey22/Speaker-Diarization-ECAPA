import os

uri = "hindi_chunk_22"
rttm_path = f"dataset/rttm/{uri}.rttm"

print(f"Checking for RTTM at: {os.path.abspath(rttm_path)}")

if os.path.exists(rttm_path):
    print("RTTM file exists!")
    with open(rttm_path, 'r') as f:
        first_line = f.readline()
        print(f"First line of RTTM: {first_line.strip()}")
        
        parts = first_line.split()
        if len(parts) > 1:
            rttm_uri = parts[1]
            if rttm_uri == uri:
                print(f"URI Match: '{rttm_uri}' matches '{uri}'")
            else:
                print(f"URI MISMATCH: RTTM says '{rttm_uri}' but protocol expects '{uri}'")
else:
    print("RTTM file NOT found at that path!")

#It checks whether an RTTM file exists, opens it, and verifies that the filename and the RTTM’s internal URI match.