import { Stack } from 'expo-router';

export default function StackLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      <Stack.Screen name="index" />
      <Stack.Screen name="Login" />
      <Stack.Screen name="SignUp" />
      <Stack.Screen name="verify-code" options={{ presentation: 'modal' }} />
      <Stack.Screen name="ForgotPassword" options={{ presentation: 'modal' }} />
      <Stack.Screen name="verify-reset" options={{ presentation: 'modal' }} />
    </Stack>
  );
}