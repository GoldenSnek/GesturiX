import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Alert,
} from 'react-native';
import React, { useState } from 'react';
import { supabase } from '../../src/supabaseClient';
import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import { router } from 'expo-router';
import { Eye, EyeOff } from 'lucide-react-native';

WebBrowser.maybeCompleteAuthSession();

const redirectUri = AuthSession.makeRedirectUri({
  path: 'oauth-callback',
});

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
  return null;
};


const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    const cleanEmail = email.trim().toLowerCase();
    const cleanPassword = password.trim();

    if (!cleanEmail || !cleanPassword) {
      Alert.alert('Error', 'Please fill in both fields');
      return;
    }

    setLoading(true);

    const { data, error } = await supabase.auth.signInWithPassword({
      email: cleanEmail,
      password: cleanPassword,
    });

    if (error) {
      Alert.alert('Login Failed', error.message);
      setLoading(false);
      return;
    }

    const user = data?.user;
    if (!user) {
      Alert.alert('Error', 'No user data returned from Supabase.');
      setLoading(false);
      return;
    }

    const { data: existingProfile, error: checkError } = await supabase
      .from('profiles')
      .select('id')
      .eq('id', user.id)
      .maybeSingle();

    if (checkError) console.error('Error checking profile:', checkError);

    if (!existingProfile) {
      const { error: insertError } = await supabase.from('profiles').insert({
        id: user.id,
        email: user.email,
        display_name: '',
      });
      if (insertError) console.error('Error creating profile:', insertError);
    }

    Alert.alert('Success', 'Login successful!');
    setLoading(false);
    router.replace('/(tabs)/translate');
  };

  const handleGoogleLogin = async () => {
    setLoading(true);

    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: redirectUri,
        skipBrowserRedirect: true,
      },
    });

    if (error) {
      Alert.alert('Google Sign In Failed', error.message);
      setLoading(false);
      return;
    }

    const authUrl = data?.url;
    if (!authUrl) {
      Alert.alert('Error', 'No authorization URL returned.');
      setLoading(false);
      return;
    }

    const result = await WebBrowser.openAuthSessionAsync(authUrl, redirectUri);
    
    if (result.type !== 'success' || !result.url) {
      setLoading(false);
      return;
    }

    const parsed = parseUrlForTokens(result.url); 

    if (parsed?.access_token) {
        await supabase.auth.setSession({
            access_token: parsed.access_token,
            refresh_token: parsed.refresh_token ?? '',
        });
        router.replace('/(tabs)/translate');
    }

    setLoading(false);
  };


  return (
    <ImageBackground
      source={require('../../assets/images/LoginSignUpBG.png')}
      className="flex-1 justify-center items-center p-8"
      resizeMode="cover"
    >
      <View className="absolute inset-0 bg-black opacity-40" />

      <View className="relative w-full max-w-sm p-8 rounded-3xl bg-white/80">
        <Text className="text-4xl font-bold text-black mb-10 text-center">
          Welcome Back
        </Text>

        <TextInput
          className="w-full border-2 border-accent rounded-lg p-4 mb-4 text-black text-lg font-bold bg-neutral"
          placeholder="Email"
          placeholderTextColor="#444444"
          autoCapitalize="none"
          value={email}
          onChangeText={setEmail}
          editable={!loading}
        />

        <View className="w-full border-2 border-accent rounded-lg mb-4 bg-neutral flex-row items-center">
          <TextInput
            className="flex-1 p-4 text-black text-lg font-bold"
            placeholder="Password"
            placeholderTextColor="#444444"
            secureTextEntry={!showPassword}
            value={password}
            onChangeText={setPassword}
            editable={!loading}
          />
          <TouchableOpacity
            onPress={() => setShowPassword(!showPassword)}
            style={{ paddingRight: 16 }}
            disabled={loading}
          >
            {showPassword ? (
              <EyeOff color="#0d1b2fff" size={22} />
            ) : (
              <Eye color="#0d1b2fff" size={22} />
            )}
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          onPress={handleLogin}
          className="w-full bg-accent rounded-full py-4 items-center mt-4"
          disabled={loading}
        >
          <Text className="text-white text-lg font-bold">
            {loading && email && password ? 'Logging In...' : 'Log In'}
          </Text>
        </TouchableOpacity>
        
        <View style={styles.separatorContainer}>
            <View style={styles.separatorLine} />
            <Text style={styles.separatorText} className="text-gray-500 mx-2">OR</Text>
            <View style={styles.separatorLine} />
        </View>

        <TouchableOpacity
          onPress={handleGoogleLogin}
          className="w-full bg-red-500 rounded-full py-4 items-center mt-4"
          disabled={loading}
        >
          <Text className="text-white text-lg font-bold">
            {loading ? 'Signing In with Google...' : 'Log In with Google'}
          </Text>
        </TouchableOpacity>


        <View className="mt-6 flex-row items-center justify-center">
          <Text className="text-black">Don't have an account? </Text>
          <TouchableOpacity onPress={() => router.replace('/(stack)/SignUp')}>
            <Text className="text-highlight font-bold">Sign Up</Text>
          </TouchableOpacity>
        </View>
      </View>
    </ImageBackground>
  );
};

export default Login;

const styles = StyleSheet.create({
    separatorContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: 20,
    },
    separatorLine: {
        flex: 1,
        height: 1,
        backgroundColor: '#D1D5DB',
    },
    separatorText: {
        fontSize: 14,
        fontWeight: 'bold',
    }
});