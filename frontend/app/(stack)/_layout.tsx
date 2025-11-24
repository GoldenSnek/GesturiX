import { Stack } from 'expo-router';

export default function StackLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="Login" />
      <Stack.Screen name="SignUp" />
      {/* Add this line: */}
      <Stack.Screen name="verify-code" options={{ presentation: 'modal' }} />
    </Stack>
  );
}