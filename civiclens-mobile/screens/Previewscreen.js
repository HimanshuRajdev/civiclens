import { View, Text, Image, TouchableOpacity, StyleSheet, ScrollView, Linking } from 'react-native';

const SEVERITY_COLORS = { High: '#EF4444', Medium: '#F59E0B', Low: '#22C55E' };

export default function PreviewScreen({ route, navigation }) {
  const { imageUri, result, lat, lng } = route.params;
  const { detection, complaint, filename, image_url } = result;

  const severityColor = SEVERITY_COLORS[detection.severity] || '#94A3B8';
  const showSeverity = detection.severity && detection.severity !== "None";
  const isFlagged = detection.flagged;

  const openMaps = () => {
    const url = `https://www.google.com/maps?q=${lat},${lng}`;
    Linking.openURL(url);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      
      <Image source={{ uri: imageUri }} style={styles.image} resizeMode="cover" />

      {/* Detection Header */}
      <View style={styles.detectionRow}>
        <Text style={styles.detectionClass}>
          {detection.class.replace(/_/g, ' ').toUpperCase()}
        </Text>

        {showSeverity && (
          <View style={[styles.severityBadge, { borderColor: severityColor }]}>
            <Text style={[styles.severityText, { color: severityColor }]}>
              {detection.severity} Priority
            </Text>
          </View>
        )}
      </View>

      {/* No Issue Message */}
      {!isFlagged && (
        <View style={styles.noIssueBox}>
          <Text style={styles.noIssueText}>
            ✅ No issue detected. Complaint submission is disabled.
          </Text>
        </View>
      )}

      {/* Duplicate Warning */}
      {result.duplicate && result.duplicate.is_duplicate && (
        <View style={styles.duplicateBox}>
          <Text style={styles.duplicateText}>
            Similar issue already reported {result.duplicate.distance_m}m away 
            (ID: {result.duplicate.existing_id})
          </Text>
        </View>
      )}

      {/* Maps Button */}
      {lat !== 0 && (
        <TouchableOpacity style={styles.mapsBtn} onPress={openMaps}>
          <Text style={styles.mapsBtnText}>View Location on Google Maps</Text>
        </TouchableOpacity>
      )}

      {/* Confidence Card */}
      <View style={styles.card}>
        <Text style={styles.cardLabel}>AI CONFIDENCE</Text>
        <View style={styles.progressBar}>
          <View
            style={[
              styles.progressFill,
              { width: `${Math.round(detection.confidence * 100)}%` }
            ]}
          />
        </View>
        <Text style={styles.confidenceText}>
          {Math.round(detection.confidence * 100)}%
        </Text>
      </View>

      {/* Complaint Card */}
      <View style={styles.card}>
        <Text style={styles.cardLabel}>AUTO-GENERATED COMPLAINT</Text>
        <Text style={styles.complaintTitle}>{complaint.title}</Text>
        <Text style={styles.complaintDesc}>{complaint.description}</Text>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Department</Text>
          <Text style={styles.infoValue}>{complaint.department}</Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Action Required</Text>
          <Text style={styles.infoValue}>{complaint.action_required}</Text>
        </View>
      </View>

      {/* Submit Button */}
      <TouchableOpacity
        style={[
          styles.submitBtn,
          { opacity: isFlagged ? 1 : 0.4 }
        ]}
        disabled={!isFlagged}
        onPress={() => {
          if (!isFlagged) return;
          navigation.navigate('Form', {
            filename,
            image_url,
            imageUri,
            detection,
            complaint,
            lat,
            lng
          });
        }}
      >
        <Text style={styles.submitBtnText}>
          {isFlagged
            ? (detection.class === "unknown"
                ? "Submit for Review"
                : "Submit Complaint")
            : "No Issue Detected"}
        </Text>
      </TouchableOpacity>

      {/* Retake */}
      <TouchableOpacity
        style={styles.retakeBtn}
        onPress={() => navigation.navigate('Camera')}
      >
        <Text style={styles.retakeBtnText}>Retake Photo</Text>
      </TouchableOpacity>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A' },
  content: { padding: 20, paddingBottom: 40 },
  image: { width: '100%', height: 220, borderRadius: 16, marginBottom: 16 },

  detectionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16
  },

  detectionClass: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '700'
  },

  severityBadge: {
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderWidth: 1
  },

  severityText: {
    fontSize: 12,
    fontWeight: '700'
  },

  noIssueBox: {
    backgroundColor: '#052e16',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#22C55E'
  },

  noIssueText: {
    color: '#22C55E',
    fontSize: 13,
    fontWeight: '600'
  },

  duplicateBox: {
    backgroundColor: '#451a03',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#f59e0b'
  },

  duplicateText: {
    color: '#fbbf24',
    fontSize: 13,
    lineHeight: 18
  },

  mapsBtn: {
    backgroundColor: '#1E293B',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#38BDF8'
  },

  mapsBtnText: {
    color: '#38BDF8',
    fontSize: 14,
    fontWeight: '700'
  },

  card: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#334155'
  },

  cardLabel: {
    color: '#64748B',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    marginBottom: 10
  },

  progressBar: {
    height: 6,
    backgroundColor: '#334155',
    borderRadius: 3,
    marginBottom: 6
  },

  progressFill: {
    height: 6,
    backgroundColor: '#38BDF8',
    borderRadius: 3
  },

  confidenceText: {
    color: '#38BDF8',
    fontSize: 13,
    fontWeight: '700'
  },

  complaintTitle: {
    color: '#F8FAFC',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8
  },

  complaintDesc: {
    color: '#94A3B8',
    fontSize: 14,
    lineHeight: 20,
    marginBottom: 16
  },

  infoRow: {
    borderTopWidth: 1,
    borderTopColor: '#334155',
    paddingTop: 12,
    marginTop: 12
  },

  infoLabel: {
    color: '#64748B',
    fontSize: 12,
    marginBottom: 4
  },

  infoValue: {
    color: '#F8FAFC',
    fontSize: 14
  },

  submitBtn: {
    backgroundColor: '#38BDF8',
    borderRadius: 16,
    paddingVertical: 18,
    alignItems: 'center',
    marginBottom: 12
  },

  submitBtnText: {
    color: '#0F172A',
    fontSize: 17,
    fontWeight: '700'
  },

  retakeBtn: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#334155'
  },

  retakeBtnText: {
    color: '#94A3B8',
    fontSize: 16,
    fontWeight: '600'
  }
});