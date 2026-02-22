import { View, Text, TouchableOpacity, StyleSheet, StatusBar } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function HomeScreen({ navigation }) {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <View style={styles.header}>
        <Text style={styles.logo}>CivicLens</Text>
        <Text style={styles.tagline}>AI-Powered City Issue Reporting</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>See a problem?</Text>
        <Text style={styles.cardSub}>
          Take a photo and our AI will detect the issue, fill in the details, and route it to the right department instantly.
        </Text>
      </View>

      <View style={styles.buttons}>
        <TouchableOpacity style={styles.primaryBtn} onPress={() => navigation.navigate('Camera')}>
        <Text style={styles.primaryBtnText}>Report an Issue</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryBtn} onPress={() => navigation.navigate('MyReports')}>
        <Text style={styles.secondaryBtnText}>My Reports</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.secondaryBtn} onPress={() => navigation.navigate('Map')}>
        <Text style={styles.secondaryBtnText}>View Incident Map</Text>
        </TouchableOpacity>
`````</View>

      <View style={styles.stats}>
        <View style={styles.statItem}>
          <Text style={styles.statNum}>6</Text>
          <Text style={styles.statLabel}>Issue Types</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statNum}>AI</Text>
          <Text style={styles.statLabel}>Auto-Detection</Text>
        </View>
        <View style={styles.statDivider} />
        <View style={styles.statItem}>
          <Text style={styles.statNum}>Live</Text>
          <Text style={styles.statLabel}>Dashboard</Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', padding: 24 },
  header: { alignItems: 'center', marginTop: 32, marginBottom: 40 },
  logo: { fontSize: 32, fontWeight: '900', color: '#F8FAFC' },
  tagline: { fontSize: 14, color: '#64748B', marginTop: 6 },
  card: { backgroundColor: '#1E293B', borderRadius: 20, padding: 24, marginBottom: 32, borderWidth: 1, borderColor: '#334155' },
  cardTitle: { fontSize: 22, fontWeight: '700', color: '#F8FAFC', marginBottom: 10 },
  cardSub: { fontSize: 15, color: '#94A3B8', lineHeight: 22 },
  buttons: { gap: 12, marginBottom: 40 },
  primaryBtn: { backgroundColor: '#38BDF8', borderRadius: 16, paddingVertical: 18, alignItems: 'center' },
  primaryBtnText: { color: '#0F172A', fontSize: 17, fontWeight: '700' },
  secondaryBtn: { backgroundColor: '#1E293B', borderRadius: 16, paddingVertical: 18, alignItems: 'center', borderWidth: 1, borderColor: '#334155' },
  secondaryBtnText: { color: '#94A3B8', fontSize: 17, fontWeight: '600' },
  stats: { flexDirection: 'row', justifyContent: 'space-around', backgroundColor: '#1E293B', borderRadius: 16, padding: 20, borderWidth: 1, borderColor: '#334155' },
  statItem: { alignItems: 'center' },
  statNum: { fontSize: 20, fontWeight: '700', color: '#38BDF8' },
  statLabel: { fontSize: 11, color: '#64748B', marginTop: 4 },
  statDivider: { width: 1, backgroundColor: '#334155' },
});