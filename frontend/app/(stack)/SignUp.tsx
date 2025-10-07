import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import React, { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';
import uuid from 'react-native-uuid';
import { Eye, EyeOff, Camera } from 'lucide-react-native';
import * as Linking from 'expo-linking';

WebBrowser.maybeCompleteAuthSession();
global.Buffer = global.Buffer || Buffer;

// 👇 Use a redirect URI that includes `--/oauth-callback`
const redirectUri = AuthSession.makeRedirectUri({
  path: 'oauth-callback',
});

const SignUp: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // 🧩 Pick image
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.6,
    });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  // 🧩 Upload avatar to Supabase
  const uploadAvatar = async (fileUri: string, userId: string) => {
    try {
      const fileExt = fileUri.split('.').pop();
      const fileName = `${uuid.v4()}.${fileExt}`;
      const filePath = `${userId}/${fileName}`;

      const base64 = await FileSystem.readAsStringAsync(fileUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const fileBytes = Buffer.from(base64, 'base64');

      const { error: uploadError } = await supabase.storage
        .from('avatars')
        .upload(filePath, fileBytes, {
          contentType: 'image/jpeg',
          upsert: true,
        });

      if (uploadError) throw uploadError;
      return filePath;
    } catch (err: any) {
      console.error('Upload error:', err.message);
      return null;
    }
  };

  // 🧩 Regular Sign-Up
  const handleSignUp = async () => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();
    const cleanUsername = username.trim();

    if (!cleanEmail || !cleanPassword || !cleanUsername) {
      Alert.alert('Error', 'Please fill in all required fields.');
      return;
    }
    if (cleanUsername.length < 3) {
      Alert.alert('Error', 'Username must be at least 3 characters.');
      return;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(cleanEmail)) {
      Alert.alert('Error', 'Please enter a valid email address.');
      return;
    }

    setLoading(true);

    try {
      const { data, error } = await supabase.auth.signUp({
        email: cleanEmail,
        password: cleanPassword,
      });

      if (error) {
        console.error('Supabase signup error:', error);
        Alert.alert('Sign Up Failed', error.message);
        setLoading(false);
        return;
      }

      const user = data.user;
      if (!user) {
        setLoading(false);
        Alert.alert('Error', 'User creation failed.');
        return;
      }

      // Upload avatar if chosen
      let photoUrl = null;
      if (image) photoUrl = await uploadAvatar(image, user.id);

      // Upsert profile
      const { error: upsertError } = await supabase.from('profiles').upsert({
        id: user.id,
        username: cleanUsername,
        email: cleanEmail,
        photo_url: photoUrl,
        updated_at: new Date().toISOString(),
      });

      if (upsertError) throw upsertError;

      await supabase.auth.signInWithPassword({
        email: cleanEmail,
        password: cleanPassword,
      });

      Alert.alert('Success', 'Account created successfully!');
      router.replace('/(stack)/LandingPage');
    } catch (err: any) {
      console.error('Signup error:', err.message);
      Alert.alert('Error', 'Unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  // 🧩 Helper for parsing tokens
  const parseUrlForTokens = (url: string) => {
    if (!url) return null;
    const hashIndex = url.indexOf('#');
    if (hashIndex !== -1) {
      const hash = url.substring(hashIndex + 1);
      const params = Object.fromEntries(new URLSearchParams(hash));
      return {
        access_token: params['access_token'] ?? null,
        refresh_token: params['refresh_token'] ?? null,
      };
    }
    const qIndex = url.indexOf('?');
    if (qIndex !== -1) {
      const query = url.substring(qIndex + 1);
      const params = Object.fromEntries(new URLSearchParams(query));
      if (params['code']) return { code: params['code'] };
    }
    return null;
  };

  // 🧩 Google OAuth Sign-Up
  const handleGoogleSignUp = async () => {
  setLoading(true);

  // 1. Initiate the OAuth Flow to get the URL
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: 'google',
    options: {
      redirectTo: redirectUri,
      skipBrowserRedirect: true, // Crucial for getting the URL to pass to WebBrowser
    },
  });

  if (error) {
    Alert.alert('Google Sign Up Failed', error.message);
    setLoading(false);
    return;
  }

  const authUrl = data?.url;
  if (!authUrl) {
    Alert.alert('Error', 'No authorization URL returned.');
    setLoading(false);
    return;
  }

  // 2. Open the browser and wait for the redirect
  const result = await WebBrowser.openAuthSessionAsync(authUrl, redirectUri);

  if (result.type !== 'success' || !result.url) {
    console.warn('OAuth cancelled or no redirect URL.');
    setLoading(false);
    return;
  }

  // 3. Manually parse the URL for tokens (Implicit Flow)
  const parsed = parseUrlForTokens(result.url); // Use your existing helper

  if (parsed?.access_token) {
    console.log('✅ Setting session from tokens (Implicit Flow)...');
    try {
      // Set the session with the tokens from the URL fragment
      const { data: sessionData, error: setErr } = await supabase.auth.setSession({
        access_token: parsed.access_token,
        refresh_token: parsed.refresh_token ?? '',
      });

      if (setErr) throw setErr;

      const user = sessionData?.session?.user;
      if (user) {
        // Your existing profile creation logic (keep this)
        const username =
          user.user_metadata?.full_name ??
          (user.email ? user.email.split('@')[0] : 'user_' + user.id.slice(0, 6));

        await supabase.from('profiles').upsert({
          id: user.id,
          email: user.email,
          username,
          photo_url: user.user_metadata?.avatar_url ?? null,
          updated_at: new Date().toISOString(),
        });

        router.replace('/(tabs)/translate');
      }
    } catch (err: any) {
      console.error('Session/Profile error:', err.message);
      Alert.alert('Sign In Failed', 'Could not establish session or create profile.');
    }
  } else {
    // Fallback if no tokens were found in the fragment
    Alert.alert('Sign In Failed', 'Session data missing from redirect.');
  }

  setLoading(false);
};
  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'android' ? 'padding' : undefined}
      className="flex-1"
    >
      <ImageBackground
        source={require('../../assets/images/LoginSignUpBG.png')}
        className="flex-1 justify-center items-center"
        resizeMode="cover"
      >
        <View className="absolute inset-0 bg-black opacity-40" />
        <ScrollView
          contentContainerStyle={{ padding: 50, paddingBottom: 80 }}
          showsVerticalScrollIndicator={false}
        >
          <View className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80">
            <Text className="text-4xl font-bold text-black mb-6 text-center">
              Create Account
            </Text>

            <TouchableOpacity
              onPress={pickImage}
              className="self-center mb-8 items-center justify-center"
            >
              <View
                style={{
                  width: 130,
                  height: 130,
                  borderRadius: 65,
                  borderWidth: 3,
                  borderColor: '#ed8b1cff',
                  backgroundColor: '#E0F2F1',
                  alignItems: 'center',
                  justifyContent: 'center',
                  shadowColor: '#000',
                  shadowOpacity: 0.25,
                  shadowRadius: 5,
                  elevation: 5,
                }}
              >
                {image ? (
                  <Image
                    source={{ uri: image }}
                    style={{ width: 124, height: 124, borderRadius: 62 }}
                  />
                ) : (
                  <>
                    <Camera color="#0d1b2fff" size={32} />
                    <Text className="text-accent font-bold mt-2">Add Photo</Text>
                  </>
                )}
              </View>
            </TouchableOpacity>

            {/* Inputs */}
            <TextInput
              className="w-full border-2 border-accent rounded-lg p-4 mb-3 text-black text-lg font-bold bg-neutral"
              placeholder="Username"
              value={username}
              onChangeText={setUsername}
              placeholderTextColor="#444"
            />
            <TextInput
              className="w-full border-2 border-accent rounded-lg p-4 mb-3 text-black text-lg font-bold bg-neutral"
              placeholder="Email"
              value={email}
              onChangeText={setEmail}
              placeholderTextColor="#444"
              autoCapitalize="none"
            />
            <View className="w-full border-2 border-accent rounded-lg mb-4 bg-neutral flex-row items-center">
              <TextInput
                className="flex-1 p-4 text-black text-lg font-bold"
                placeholder="Password"
                placeholderTextColor="#444"
                secureTextEntry={!showPassword}
                value={password}
                onChangeText={setPassword}
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                style={{ paddingRight: 16 }}
              >
                {showPassword ? (
                  <EyeOff color="#0d1b2fff" size={22} />
                ) : (
                  <Eye color="#0d1b2fff" size={22} />
                )}
              </TouchableOpacity>
            </View>

            {/* Buttons */}
            <TouchableOpacity
              onPress={handleSignUp}
              disabled={loading}
              className="w-full bg-accent rounded-full py-4 items-center mt-2"
            >
              <Text className="text-white text-lg font-bold">
                {loading ? 'Creating Account...' : 'Sign Up'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={handleGoogleSignUp}
              className="w-full bg-red-500 rounded-full py-4 items-center mt-4"
            >
              <Text className="text-white text-lg font-bold">
                Sign Up with Google
              </Text>
            </TouchableOpacity>

            <View className="mt-6 flex-row items-center justify-center">
              <Text className="text-black">Already have an account? </Text>
              <TouchableOpacity onPress={() => router.replace('/(stack)/Login')}>
                <Text className="text-highlight font-bold">Log In</Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </ImageBackground>
    </KeyboardAvoidingView>
  );
};

export default SignUp;

const styles = StyleSheet.create({});
