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
import { supabase } from '../../src/supabaseClient';
import { router } from 'expo-router';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
import uuid from 'react-native-uuid';
import { Eye, EyeOff, Camera } from 'lucide-react-native'; // <— icons

global.Buffer = global.Buffer || Buffer;
const redirectUrl = AuthSession.makeRedirectUri();

const SignUp: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // 🔹 Pick profile image from gallery
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.6,
    });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  // 🔹 Upload avatar to Supabase Storage
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

      return filePath; // store relative path
    } catch (err: any) {
      console.error('Upload error:', err.message);
      return null;
    }
  };

  // 🔹 Sign Up Logic (let DB trigger create profile)
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
      console.log('Creating user...');
      const { data, error } = await supabase.auth.signUp({
        email: cleanEmail,
        password: cleanPassword,
        options: {
          data: {
            username: cleanUsername,
          }
        }
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

      // 🧱 Upload avatar if chosen (wait for auto profile creation, then update)
      if (image) {
        let photoUrl = await uploadAvatar(image, user.id);
        if (photoUrl) {
          // Update the profile with photo_url (allowed by RLS after profile exists)
          await supabase.from('profiles').update({
            photo_url: photoUrl,
            updated_at: new Date().toISOString(),
          }).eq('id', user.id);
        }
      }

      setLoading(false);

      // Replace this with your app's desired post-signup flow (e.g. verification, login prompt, etc.)
      Alert.alert('Success', 'Account created successfully!');
      router.replace('/(stack)/LandingPage');
    } catch (err: any) {
      console.error('Signup process error:', err.message);
      Alert.alert('Error', 'An unexpected error occurred.');
      setLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: redirectUrl },
    });
    if (error) Alert.alert('Google Sign Up Failed', error.message);
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      className="flex-1"
    >
      <ImageBackground
        source={require('../../assets/images/LoginSignUpBG.png')}
        className="flex-1 justify-center items-center"
        resizeMode="cover"
      >
        <View className="absolute inset-0 bg-black opacity-40" />

        <ScrollView
          contentContainerStyle={{ padding: 24, paddingBottom: 80 }}
          showsVerticalScrollIndicator={false}
        >
          <View className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80">
            <Text className="text-4xl font-bold text-black mb-6 text-center">Create Account</Text>

            {/* Profile Image Picker (Enhanced) */}
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
                  borderColor: '#4DB6AC',
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
                    <Camera color="#0D47A1" size={32} />
                    <Text className="text-accent font-bold mt-2">Add Photo</Text>
                  </>
                )}
              </View>
            </TouchableOpacity>

            {/* Username */}
            <TextInput
              className="w-full border-2 border-accent rounded-lg p-4 mb-3 text-black text-lg font-bold bg-neutral"
              placeholder="Username"
              placeholderTextColor="#444444"
              value={username}
              onChangeText={setUsername}
            />

            {/* Email */}
            <TextInput
              className="w-full border-2 border-accent rounded-lg p-4 mb-3 text-black text-lg font-bold bg-neutral"
              placeholder="Email"
              placeholderTextColor="#444444"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
            />

            {/* Password with Eye Toggle */}
            <View className="w-full border-2 border-accent rounded-lg mb-4 bg-neutral flex-row items-center">
              <TextInput
                className="flex-1 p-4 text-black text-lg font-bold"
                placeholder="Password"
                placeholderTextColor="#444444"
                secureTextEntry={!showPassword}
                value={password}
                onChangeText={setPassword}
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                style={{ paddingRight: 16 }}
              >
                {showPassword ? (
                  <EyeOff color="#0D47A1" size={22} />
                ) : (
                  <Eye color="#0D47A1" size={22} />
                )}
              </TouchableOpacity>
            </View>

            {/* Email Sign Up */}
            <TouchableOpacity
              onPress={handleSignUp}
              disabled={loading}
              className="w-full bg-accent rounded-full py-4 items-center mt-2"
            >
              <Text className="text-white text-lg font-bold">
                {loading ? 'Creating Account...' : 'Sign Up'}
              </Text>
            </TouchableOpacity>

            {/* Google Sign Up */}
            <TouchableOpacity
              onPress={handleGoogleSignUp}
              className="w-full bg-red-500 rounded-full py-4 items-center mt-4"
            >
              <Text className="text-white text-lg font-bold">Sign Up with Google</Text>
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
