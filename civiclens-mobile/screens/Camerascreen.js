import { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import axios from 'axios';
import { API_URL } from '../config';

export default function CameraScreen({ navigation }) {
  const [loading, setLoading] = useState(false);

  const getLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return { lat: 0, lng: 0 };
      const loc = await Location.getCurrentPositionAsync({});
      return { lat: loc.coords.latitude, lng: loc.coords.longitude };
    } catch {
      return { lat: 0, lng: 0 };
    }
  };

  const handleImage = async (uri) => {
    setLoading(true);
    try {
      const { lat, lng } = await getLocation();
      const formData = new FormData();
      formData.append('file', { uri, name: 'photo.jpg', type: 'image/jpeg' });
      const response = await axios.post(
        `${API_URL}/report?lat=${lat}&lng=${lng}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      navigation.navigate('Preview', { imageUri: uri, result: response.data, lat, lng });
    } catch (error) {
      Alert.alert('Detection Failed', error.response?.data?.detail || 'No civic issue detected. Try a clearer photo.');
    } finally {
      setLoading(false);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') { Alert.alert('Permission needed', 'Camera access is required.'); return; }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) handleImage(result.assets[0].uri);
  };

  const pickPhoto = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
    if (!result.canceled) handleImage(result.assets[0].uri);
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size={48} color="#38BDF8" />
        <Text style={styles.loadingText}>AI is analyzing your image...</Text>
        <Text style={styles.loadingSub}>Detecting issue type, severity and department</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.hero}>
        <Text style={styles.heroTitle}>Capture the Issue</Text>
        <Text style={styles.heroSub}>Take a clear photo of the civic problem. Our AI will automatically detect and classify it.</Text>
      </View>
      <View style={styles.options}>
        <TouchableOpacity style={styles.optionCard} onPress={takePhoto}>
          <Text style={styles.optionTitle}>Take Photo</Text>
          <Text style={styles.optionSub}>Use your camera</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.optionCard} onPress={pickPhoto}>
          <Text style={styles.optionTitle}>Upload Photo</Text>
          <Text style={styles.optionSub}>From your gallery</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.tipBox}>
        <Text style={styles.tipTitle}>Tips for best results</Text>
        <Text style={styles.tip}>Get close to the issue</Text>
        <Text style={styles.tip}>Make sure it is well lit</Text>
        <Text style={styles.tip}>Keep the camera steady</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', padding: 24 },
  loadingContainer: { flex: 1, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center', padding: 24 },
  loadingText: { color: '#F8FAFC', fontSize: 18, fontWeight: '700', marginTop: 24, textAlign: 'center' },
  loadingSub: { color: '#64748B', fontSize: 14, marginTop: 8, textAlign: 'center' },
  hero: { alignItems: 'center', marginTop: 20, marginBottom: 40 },
  heroTitle: { fontSize: 26, fontWeight: '700', color: '#F8FAFC', marginTop: 16 },
  heroSub: { fontSize: 14, color: '#94A3B8', textAlign: 'center', marginTop: 8, lineHeight: 20 },
  options: { flexDirection: 'row', gap: 12, marginBottom: 32 },
  optionCard: { flex: 1, backgroundColor: '#1E293B', borderRadius: 20, padding: 24, alignItems: 'center', borderWidth: 1, borderColor: '#334155' },
  optionTitle: { color: '#F8FAFC', fontSize: 16, fontWeight: '700' },
  optionSub: { color: '#64748B', fontSize: 12, marginTop: 4 },
  tipBox: { backgroundColor: '#1E293B', borderRadius: 16, padding: 20, borderWidth: 1, borderColor: '#334155' },
  tipTitle: { color: '#38BDF8', fontSize: 14, fontWeight: '700', marginBottom: 10 },
  tip: { color: '#94A3B8', fontSize: 13, marginBottom: 4 },
});