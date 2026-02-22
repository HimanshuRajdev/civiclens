import { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity } from 'react-native';
import MapView, { Marker, Callout } from 'react-native-maps';
import axios from 'axios';
import { API_URL } from '../config';

const SEVERITY_COLORS = { High: '#EF4444', Medium: '#F59E0B', Low: '#22C55E' };

export default function MapScreen() {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetchComplaints();
  }, []);

  const fetchComplaints = async () => {
    try {
      const response = await axios.get(`${API_URL}/complaints`);
      // Only show complaints that have real location data
      const withLocation = response.data.filter(c => c.lat !== 0 && c.lng !== 0);
      setComplaints(withLocation);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size={48} color="#38BDF8" />
        <Text style={styles.loadingText}>Loading incidents...</Text>
      </View>
    );
  }

  const initialRegion = complaints.length > 0
    ? {
        latitude: complaints[0].lat,
        longitude: complaints[0].lng,
        latitudeDelta: 0.05,
        longitudeDelta: 0.05,
      }
    : {
        latitude: 43.0731,
        longitude: -89.4012,
        latitudeDelta: 0.1,
        longitudeDelta: 0.1,
      };

  return (
    <View style={styles.container}>
      <MapView
        style={styles.map}
        initialRegion={initialRegion}
        showsUserLocation
        showsMyLocationButton
      >
        {complaints.map((complaint) => (
          <Marker
            key={complaint.id}
            coordinate={{ latitude: complaint.lat, longitude: complaint.lng }}
            pinColor={SEVERITY_COLORS[complaint.severity] || '#94A3B8'}
            onPress={() => setSelected(complaint)}
          >
            <Callout>
              <View style={styles.callout}>
                <Text style={styles.calloutType}>{complaint.issue_type?.replace(/_/g, ' ').toUpperCase()}</Text>
                <Text style={styles.calloutId}>{complaint.id}</Text>
                <Text style={styles.calloutStatus}>{complaint.status}</Text>
              </View>
            </Callout>
          </Marker>
        ))}
      </MapView>

      {/* Legend */}
      <View style={styles.legend}>
        <Text style={styles.legendTitle}>Incidents: {complaints.length}</Text>
        <View style={styles.legendRow}>
          <View style={[styles.dot, { backgroundColor: '#EF4444' }]} />
          <Text style={styles.legendLabel}>High</Text>
          <View style={[styles.dot, { backgroundColor: '#F59E0B' }]} />
          <Text style={styles.legendLabel}>Medium</Text>
          <View style={[styles.dot, { backgroundColor: '#22C55E' }]} />
          <Text style={styles.legendLabel}>Low</Text>
        </View>
      </View>

      {/* Selected complaint detail card */}
      {selected && (
        <View style={styles.detailCard}>
          <TouchableOpacity style={styles.closeBtn} onPress={() => setSelected(null)}>
            <Text style={styles.closeBtnText}>X</Text>
          </TouchableOpacity>
          <Text style={styles.detailType}>{selected.issue_type?.replace(/_/g, ' ').toUpperCase()}</Text>
          <Text style={styles.detailId}>{selected.id}</Text>
          <Text style={styles.detailDesc} numberOfLines={2}>{selected.description}</Text>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Department</Text>
            <Text style={styles.detailValue}>{selected.department}</Text>
          </View>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Status</Text>
            <Text style={[styles.detailValue, { color: SEVERITY_COLORS[selected.severity] }]}>{selected.status}</Text>
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  map: { flex: 1 },
  centered: { flex: 1, backgroundColor: '#0F172A', alignItems: 'center', justifyContent: 'center' },
  loadingText: { color: '#F8FAFC', fontSize: 16, marginTop: 16 },

  legend: {
    position: 'absolute',
    top: 16,
    left: 16,
    backgroundColor: 'rgba(15,23,42,0.9)',
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  legendTitle: { color: '#F8FAFC', fontSize: 13, fontWeight: '700', marginBottom: 6 },
  legendRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  dot: { width: 10, height: 10, borderRadius: 5 },
  legendLabel: { color: '#94A3B8', fontSize: 11, marginRight: 6 },

  detailCard: {
    position: 'absolute',
    bottom: 24,
    left: 16,
    right: 16,
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  closeBtn: { position: 'absolute', top: 12, right: 12, padding: 4 },
  closeBtnText: { color: '#64748B', fontSize: 14, fontWeight: '700' },
  detailType: { color: '#F8FAFC', fontSize: 15, fontWeight: '700', marginBottom: 2 },
  detailId: { color: '#64748B', fontSize: 11, marginBottom: 8 },
  detailDesc: { color: '#94A3B8', fontSize: 13, lineHeight: 18, marginBottom: 12 },
  detailRow: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 6 },
  detailLabel: { color: '#64748B', fontSize: 12 },
  detailValue: { color: '#F8FAFC', fontSize: 12, fontWeight: '600' },
});