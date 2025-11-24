import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Image,
  Alert, 
} from 'react-native';
import React, { useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import * as AuthSession from 'expo-auth-session';
import { supabase } from '../../src/supabaseClient';
// Removed duplicate router import
// import { router } from 'expo-router'; 
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
import uuid from 'react-native-uuid';
import { Eye, EyeOff, Camera, ChevronLeft } from 'lucide-react-native';
import Message, { MessageType } from '../../components/Message';
import { useAuth } from '../../src/AuthContext';
import { useRouter } from 'expo-router';

// 🟦 Reanimated
import Animated, { FadeInUp, FadeInDown } from 'react-native-reanimated';

global.Buffer = global.Buffer || Buffer;
const redirectUrl = AuthSession.makeRedirectUri();

const SignUp: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<MessageType>('error');
  
  // Use the updated signUp from context
  const { signUp } = useAuth();
  const router = useRouter();

  const showStatus = (msg: string, type: MessageType) => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  const showError = (msg: string) => showStatus(msg, 'error');
  const showWarning = (msg: string) => showStatus(msg, 'warning');
  const showSuccess = (msg: string) => showStatus(msg, 'success');

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.6,
    });
    if (!result.canceled) setImage(result.assets[0].uri);
  };

  const handleSignUp = async () => {
    // 1. Validate Username is present
    if (!email || !password || !username) {
      Alert.alert('Error', 'Please fill in all fields, including username.');
      return;
    }

    if (username.length < 3) {
      Alert.alert('Error', 'Username must be at least 3 characters long.');
      return;
    }

    setLoading(true); // Start loading state

    // 2. Call Supabase Sign Up with USERNAME
    const { error } = await signUp(email, password, username);

    setLoading(false); // End loading state

    if (error) {
      Alert.alert('Sign Up Failed', error.message);
    } else {
      // 3. On success, navigate to verify-code
      router.push({
        pathname: '/(stack)/verify-code',
        params: { email: email }
      });
    }
  };

  const handleGoogleSignUp = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: redirectUrl },
    });
    if (error) showError(`Google Sign Up Failed: ${error.message}`);
  };

  const handleGoBack = () => {
    if (router.canGoBack()) router.back();
    else router.replace('/(stack)/Login');
  };

  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      <TouchableOpacity
        onPress={handleGoBack}
        className="absolute top-12 left-8 p-2 rounded-full bg-white/80 z-10"
      >
        <ChevronLeft color="#1A1A1A" size={28} />
      </TouchableOpacity>

      <Message message={message} type={messageType} onClose={() => setMessage('')} />

      {/* Card animation only */}
      <Animated.View
        entering={FadeInUp.duration(700).delay(150)}
        className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80"
      >
        {/* Title */}
        <Animated.Text
          entering={FadeInDown.delay(200).duration(600)}
          className="text-4xl text-black mb-6 text-center font-audiowide"
        >
          Create Account
        </Animated.Text>

        {/* Avatar */}
        <Animated.View entering={FadeInUp.delay(250).duration(600)} className="self-center mb-8">
          <TouchableOpacity onPress={pickImage} className="items-center justify-center">
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
                  <Text className="text-accent font-fredoka-semibold mt-2">Add Photo</Text>
                </>
              )}
            </View>
          </TouchableOpacity>
        </Animated.View>

        {/* Username */}
        <Animated.View entering={FadeInUp.delay(350).duration(600)}>
          <TextInput
            className="w-full border-2 border-accent rounded-lg p-4 mb-3 text-black text-lg font-montserrat-semibold bg-neutral"
            placeholder="Username"
            placeholderTextColor="#444444"
            value={username}
            onChangeText={setUsername}
            selectionColor="#FFAB7B"
          />
        </Animated.View>

        {/* Email */}
        <Animated.View entering={FadeInUp.delay(450).duration(600)}>
          <TextInput
            className="w-full border-2 border-accent rounded-lg p-4 mb-3 text-black text-lg font-montserrat-semibold bg-neutral"
            placeholder="Email"
            placeholderTextColor="#444444"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            selectionColor="#FFAB7B"
          />
        </Animated.View>

        {/* Password */}
        <Animated.View entering={FadeInUp.delay(550).duration(600)}>
          <View className="w-full border-2 border-accent rounded-lg mb-4 bg-neutral flex-row items-center">
            <TextInput
              className="flex-1 p-4 text-black text-lg font-montserrat-semibold"
              placeholder="Password"
              placeholderTextColor="#444444"
              secureTextEntry={!showPassword}
              value={password}
              onChangeText={setPassword}
              selectionColor="#FFAB7B"
            />
            <TouchableOpacity onPress={() => setShowPassword(!showPassword)} style={{ paddingRight: 16 }}>
              {showPassword ? <EyeOff color="#0D47A1" size={22} /> : <Eye color="#0D47A1" size={22} />}
            </TouchableOpacity>
          </View>
        </Animated.View>

        {/* Sign Up Button */}
        <Animated.View entering={FadeInUp.delay(650).duration(600)} style={{ backgroundColor: 'transparent' }}>
          <TouchableOpacity
            onPress={handleSignUp}
            disabled={loading}
            className="w-full bg-accent rounded-full py-4 items-center mt-2 shadow-lg"
          >
            <Text className="text-white text-lg font-audiowide">
              {loading ? 'Creating Account...' : 'Sign Up'}
            </Text>
          </TouchableOpacity>
        </Animated.View>

        {/* Google Sign Up */}
        <Animated.View entering={FadeInUp.delay(750).duration(600)} style={{ backgroundColor: 'transparent' }}>
          <TouchableOpacity
            onPress={handleGoogleSignUp}
            className="w-full bg-red-500 rounded-full py-4 items-center mt-4 shadow-lg"
          >
            <Text className="text-white text-lg font-audiowide">Sign Up with Google</Text>
          </TouchableOpacity>
        </Animated.View>

        {/* Footer */}
        <Animated.View entering={FadeInUp.delay(850).duration(600)} className="mt-6 flex-row items-center justify-center">
          <Text className="text-black font-fredoka">Already have an account? </Text>
          <TouchableOpacity onPress={() => router.replace('/(stack)/Login')}>
            <Text className="text-highlight font-fredoka-semibold">Log In</Text>
          </TouchableOpacity>
        </Animated.View>
      </Animated.View>
    </ImageBackground>
  );
};

export default SignUp;

const styles = StyleSheet.create({});