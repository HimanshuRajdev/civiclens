import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from './screens/Homescreen';
import CameraScreen from './screens/Camerascreen';
import PreviewScreen from './screens/Previewscreen';
import FormScreen from './screens/Formscreen';
import MyReportsScreen from './screens/Myreportsscreen';
import MapScreen from './screens/Mapscreen';


const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="Camera" component={CameraScreen} />
        <Stack.Screen name="Preview" component={PreviewScreen} />
        <Stack.Screen name="Form" component={FormScreen} />
        <Stack.Screen name="MyReports" component={MyReportsScreen} />
        <Stack.Screen name="Map" component={MapScreen} />

      </Stack.Navigator>
    </NavigationContainer>
  );
}