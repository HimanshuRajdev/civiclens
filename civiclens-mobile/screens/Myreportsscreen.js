import { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, ActivityIndicator, RefreshControl } from 'react-native';
import axios from 'axios';
import { API_URL } from '../config';

const STATUS_COLORS = { 'Open': '#EF4444', 'In Progress': '#F59E0B', 'Resolved': '#22C55E' };

export default function MyReportsScreen() {
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchComplaints = async () => {
    try {
      const response = await axios.get(`${API_URL}/complaints`);
      setComplaints(response.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => { fetchComplaints(); }, []);

  const renderItem = ({ item }) => {
    const statusColor = STATUS_COLORS[item.status] || '#94A3B8';
    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardType}>{item.issue_type?.replace(/_/g, ' ').toUpperCase()}</Text>
          <View style={[styles.statusBadge, { borderColor: statusColor }]}>
            <Text style={[styles.statusText, { color: statusColor }]}>{item.status}</Text>
          </View>
        </View>
        <Text style={styles.cardId}>{item.id}</Text>
        <Text style={styles.cardDesc} numberOfLines={2}>{item.description}</Text>
        <Text style={styles.cardDate}>{new Date(item.created_at).toLocaleDateString()}</Text>
      </View>
    );
  };

  if (loading) return <View style={styles.centered}>pt<ActivityIndicator size={48} color="#38BDF8" /></View>;

  return (
    <View style={styles.container}>
      {complaints.length === 0 ? (
        <View style={styles.centered}>
          <Text style={styles.emptyText}>No reports yet</Text>
          <Text style={styles.emptySub}>Your submitted complaints will appear here</Text>
        </View>
      ) : (
        <FlatList
          data={complaints}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => { setRefreshing(true); fetchComplaints(); }} tintColor="#38BDF8" />}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A' },
  centered: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  list: { padding: 16, gap: 12 },
  card: { backgroundColor: '#1E293B', borderRadius: 16, padding: 16, borderWidth: 1, borderColor: '#334155' },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  cardType: { color: '#F8FAFC', fontSize: 13, fontWeight: '700' },
  statusBadge: { borderRadius: 20, paddingHorizontal: 10, paddingVertical: 4, borderWidth: 1 },
  statusText: { fontSize: 11, fontWeight: '700' },
  cardId: { color: '#64748B', fontSize: 11, marginBottom: 8 },
  cardDesc: { color: '#94A3B8', fontSize: 13, lineHeight: 18, marginBottom: 8 },
  cardDate: { color: '#475569', fontSize: 11 },
  emptyText: { color: '#F8FAFC', fontSize: 20, fontWeight: '700' },
  emptySub: { color: '#64748B', fontSize: 14, marginTop: 8 },
});