import { Redirect } from 'expo-router';

// This file simply redirects the root path to your login stack
export default function Index() {
  return <Redirect href="/(stack)/Login" />;
}