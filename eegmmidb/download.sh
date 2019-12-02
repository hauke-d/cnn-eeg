#wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/
cp -r physionet.org/files/eegmmidb/1.0.0/S*/*.edf .
rm -rf physionet.org
