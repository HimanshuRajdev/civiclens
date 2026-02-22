import { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView, Alert, ActivityIndicator } from 'react-native';
import axios from 'axios';
import { API_URL } from '../config';

export default function FormScreen({ route, navigation }) {
  const { filename, image_url, detection, complaint, lat, lng } = route.params;
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/submit`, {
        filename,
        image_url,
        issue_type: detection.class,
        severity: detection.severity,
        description: complaint.description + (notes ? '\n\nUser notes: ' + notes : ''),
        department: complaint.department,
        lat,
        lng,
      });
      navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
      Alert.alert('Complaint Submitted!', 'Your complaint ID is: ' + response.data.complaint_id + '\n\nRouted to ' + complaint.department);
    } catch {
      Alert.alert('Submission Failed', 'Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionTitle}>Review Your Complaint</Text>

      {[
        { label: 'Issue Type', value: detection.class.replace(/_/g, ' ').toUpperCase() },
        { label: 'Complaint Title', value: complaint.title },
        { label: 'Description', value: complaint.description },
        { label: 'Department', value: complaint.department },
        { label: 'Priority', value: detection.severity },
      ].map((field) => (
        <View style={styles.field} key={field.label}>
          <Text style={styles.label}>{field.label}</Text>
          <View style={styles.readonlyField}>
            <Text style={styles.readonlyText}>{field.value}</Text>
          </View>
        </View>
      ))}

      <View style={styles.field}>
        <Text style={styles.label}>Additional Notes (Optional)</Text>
        <TextInput
          style={styles.textInput}
          value={notes}
          onChangeText={setNotes}
          placeholder="Add any extra details..."
          placeholderTextColor="#475569"
          multiline
          numberOfLines={4}
        />
      </View>

      <TouchableOpacity style={[styles.submitBtn, loading && { opacity: 0.6 }]} onPress={handleSubmit} disabled={loading}>
        {loading ? <ActivityIndicator color="#0F172A" /> : <Text style={styles.submitBtnText}>Submit to City Authority</Text>}
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A' },
  content: { padding: 20, paddingBottom: 40 },
  sectionTitle: { color: '#F8FAFC', fontSize: 22, fontWeight: '700', marginBottom: 24 },
  field: { marginBottom: 16 },
  label: { color: '#64748B', fontSize: 12, fontWeight: '700', marginBottom: 8 },
  readonlyField: { backgroundColor: '#1E293B', borderRadius: 12, padding: 14, borderWidth: 1, borderColor: '#334155' },
  readonlyText: { color: '#F8FAFC', fontSize: 14, lineHeight: 20 },
  textInput: { backgroundColor: '#1E293B', borderRadius: 12, padding: 14, borderWidth: 1, borderColor: '#38BDF8', color: '#F8FAFC', fontSize: 14, minHeight: 100, textAlignVertical: 'top' },
  submitBtn: { backgroundColor: '#38BDF8', borderRadius: 16, paddingVertical: 18, alignItems: 'center', marginTop: 8 },
  submitBtnText: { color: '#0F172A', fontSize: 17, fontWeight: '700' },
});