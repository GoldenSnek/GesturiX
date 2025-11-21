import React, { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  Switch,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
  PanResponder,
  TextInput,
  Alert,
  Linking,
  BackHandler,
  ImageBackground,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, AntDesign } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { supabase } from '../../src/supabaseClient';
import AppHeaderProfile from '../../components/AppHeaderProfile';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
import { useTheme } from '../../src/ThemeContext';
import { useFocusEffect } from '@react-navigation/native';
import { fetchUserStatistics, getCurrentUserId } from '../../utils/supabaseApi';

global.Buffer = global.Buffer || Buffer;

interface ProfileData {
  id: string;
  username: string;
  email: string;
  created_at: string;
  photo_url?: string | null;
}

const Profile = () => {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { isDark, toggleTheme } = useTheme();

  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [isEditingUsername, setIsEditingUsername] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [usernameLoading, setUsernameLoading] = useState(false);

  // NEW: State for Supabase statistics
  const [userStats, setUserStats] = useState({
    lessons_completed: 0,
    days_streak: 0,
    practice_hours: 0,
  });

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) =>
        Math.abs(gestureState.dx) > 10,
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dx > 30) {
          router.push('/learn');
        }
      },
    })
  ).current;

  // Fetch user statistics when screen is focused
  useFocusEffect(
    React.useCallback(() => {
      let isActive = true;
      async function loadStats() {
        const userId = await getCurrentUserId();
        if (typeof userId !== 'string') return;
        const stats = await fetchUserStatistics(userId);
        if (isActive) setUserStats(stats);
      }
      loadStats();
      return () => { isActive = false; };
    }, [])
  );

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        setLoading(true);
        const {
          data: { user },
          error: userError,
        } = await supabase.auth.getUser();
        if (userError || !user) throw userError || new Error('No user logged in');

        const { data, error } = await supabase
          .from('profiles')
          .select('id, username, email, created_at, photo_url')
          .eq('id', user.id)
          .single();

        if (error) throw error;
        setProfile(data as ProfileData);
        setNewUsername(data.username);

        if (data.photo_url) {
          const { data: urlData } = supabase.storage
            .from('avatars')
            .getPublicUrl(data.photo_url);
          setPhotoUrl(urlData?.publicUrl || null);
        } else {
          setPhotoUrl(null);
        }
      } catch (err: any) {
        console.error('Error fetching profile:', err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.replace('/(stack)');
  };

  const handleSaveUsername = async () => {
    if (newUsername.trim().length < 3) {
      Alert.alert('Username must be at least 3 characters.');
      return;
    }
    setUsernameLoading(true);
    const { error } = await supabase
      .from('profiles')
      .update({ username: newUsername.trim(), updated_at: new Date().toISOString() })
      .eq('id', profile?.id);

    if (error) {
      Alert.alert('Update failed', error.message);
    } else {
      setProfile((prev) =>
        prev ? { ...prev, username: newUsername.trim() } : null
      );
      setIsEditingUsername(false);
    }
    setUsernameLoading(false);
  };

  const handleChangePhoto = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });
    if (!result.canceled) {
      try {
        const fileUri = result.assets[0].uri;
        const fileExt = fileUri.split('.').pop();
        const fileName = `${profile?.id}/${Date.now()}.${fileExt}`;
        const base64 = await FileSystem.readAsStringAsync(fileUri, { encoding: FileSystem.EncodingType.Base64 });
        const fileBytes = Buffer.from(base64, 'base64');
        const { error: uploadError } = await supabase.storage
          .from('avatars')
          .upload(fileName, fileBytes, { contentType: 'image/jpeg', upsert: true });
        if (uploadError) {
          Alert.alert('Photo upload failed.', uploadError.message);
          return;
        }
        const { data: urlData } = supabase.storage.from('avatars').getPublicUrl(fileName);
        await supabase
          .from('profiles')
          .update({ photo_url: fileName, updated_at: new Date().toISOString() })
          .eq('id', profile?.id);
        setPhotoUrl(urlData?.publicUrl || null);
      } catch (error: any) {
        Alert.alert('Photo upload failed.', error.message);
      }
    }
  };

  if (loading) {
    return (
      <View className={`flex-1 items-center justify-center ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}>
        <ActivityIndicator size="large" color="accent" />
      </View>
    );
  }

  if (!profile) {
    return (
      <View className={`flex-1 items-center justify-center ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}>
        <Text
          className={`${isDark ? 'text-secondary' : 'text-primary'}`}
          style={{ fontFamily: 'Montserrat-SemiBold' }}
        >
          No profile data found.
        </Text>
      </View>
    );
  }

  // Helper for GitHub link open
  const openGitHub = async (url: string) => {
    const supported = await Linking.canOpenURL(url);
    if (supported) await Linking.openURL(url);
    else Alert.alert('Unable to open link');
  };
  
  // NOTE: buttonSettings array has been removed.

  return (
    // The outermost container now sets the theme color (which shows through the transparent PNG)
    <View className={`flex-1 ${isDark ? 'bg-darkbg' : 'bg-secondary'}`}>
      
      {/* Replaced absolute Views/Images with a single ImageBackground component */}
      <ImageBackground
        source={require('../../assets/images/MainBG.png')}
        className="flex-1" // Ensures the ImageBackground fills the parent View
        resizeMode="cover" // Ensures the image covers the entire background
      >

        {/* Foreground Content Layer (Header and ScrollView) */}
        <View 
          className="flex-1" 
          style={{ paddingTop: insets.top }} 
          // pointerEvents="box-none" is no longer needed here as ImageBackground handles touch events
        >

          <ScrollView
            {...panResponder.panHandlers}
            className="flex-1 p-4"
            contentContainerStyle={{ paddingBottom: 150 }}
            keyboardShouldPersistTaps="handled"
          >
            {/* 🧍 Profile Section */}
            <View className="items-center mt-10 mb-8 px-6">
              {/* Profile Image Container */}
              <View className="relative items-center">
                <Image
                  source={{
                    uri:
                      photoUrl ||
                      'https://ui-avatars.com/api/?name=User&background=999&color=fff',
                  }}
                  style={{
                    width: 150,
                    height: 150,
                    borderRadius: 75,
                    borderWidth: 3,
                    borderColor: isDark ? '#FF6B00' : '#FFAB7B',
                    backgroundColor: isDark ? '#1E1E1E' : '#F8F8F8',
                  }}
                />

                {/* Edit Photo Button */}
                <TouchableOpacity
                  onPress={handleChangePhoto}
                  className={`absolute bottom-2 right-3 rounded-full p-2 shadow-lg active:opacity-80 ${
                    isDark ? 'bg-darksurface' : 'bg-white'
                  }`}
                >
                  <MaterialIcons name="photo-camera" size={22} color="#FF6B00" />
                </TouchableOpacity>
              </View>

              {/* Username Section */}
              <View className="items-center mt-5 w-[260px]">
                {!isEditingUsername ? (
                  <View className="flex-row items-center space-x-2">
                    <Text
                      className={`text-2xl ${
                        isDark ? 'text-secondary' : 'text-primary'
                      }`}
                      style={{ fontFamily: 'Fredoka-SemiBold' }}
                    >
                      {profile.username}
                    </Text>
                    <TouchableOpacity
                      onPress={() => setIsEditingUsername(true)}
                      activeOpacity={0.8}
                    >
                      <MaterialIcons name="edit" size={20} color="#FF6B00" />
                    </TouchableOpacity>
                  </View>
                ) : (
                  <View className="w-full items-center">
                    <TextInput
                      value={newUsername}
                      onChangeText={setNewUsername}
                      editable={!usernameLoading}
                      placeholder="Enter new username"
                      placeholderTextColor={isDark ? '#A8A8A8' : '#888'}
                      className={`w-full border-b-2 pb-1 text-center text-lg ${
                        isDark
                          ? 'border-accent text-secondary bg-darksurface'
                          : 'border-accent text-primary bg-white'
                      }`}
                      style={{ fontFamily: 'Montserrat-SemiBold' }}
                    />

                    {/* Uniform Action Buttons */}
                    <View className="flex-row justify-center mt-4">
                      <TouchableOpacity
                        onPress={handleSaveUsername}
                        disabled={usernameLoading}
                        className="bg-accent px-6 py-2 rounded-full active:opacity-80 mx-2"
                      >
                        <Text
                          className="text-white text-base"
                          style={{ fontFamily: 'Fredoka-SemiBold' }}
                        >
                          Save
                        </Text>
                      </TouchableOpacity>

                      <TouchableOpacity
                        onPress={() => setIsEditingUsername(false)}
                        disabled={usernameLoading}
                        className={`px-6 py-2 rounded-full border active:opacity-80 mx-2 ${
                          isDark ? 'border-neutral' : 'border-primary'
                        }`}
                      >
                        <Text
                          className={`text-base ${
                            isDark ? 'text-neutral' : 'text-primary'
                          }`}
                          style={{ fontFamily: 'Fredoka-SemiBold' }}
                        >
                          Cancel
                        </Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                )}
              </View>
            </View>

            {/* 🏆 Progress Section - UPDATED WITH LIVE DATA */}
            <Text
              className={`text-xl mb-3 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              My Progress
            </Text>

            <View className="flex-row flex-wrap justify-between mb-8">
              <View
                className={`w-[48%] rounded-2xl p-4 mb-4 items-center border border-accent shadow-md ${
                  isDark ? 'bg-darksurface' : 'bg-white'
                }`}
              >
                <Text className="text-3xl text-accent mb-1" style={{ fontFamily: 'Fredoka-SemiBold' }}>
                  {userStats.lessons_completed}
                </Text>
                <Text
                  className={`text-sm text-center capitalize ${
                    isDark ? 'text-neutral' : 'text-primary'
                  }`}
                  style={{ fontFamily: 'Montserrat-SemiBold' }}
                >
                  Lessons Completed
                </Text>
              </View>

              <View
                className={`w-[48%] rounded-2xl p-4 mb-4 items-center border border-accent shadow-md ${
                  isDark ? 'bg-darksurface' : 'bg-white'
                }`}
              >
                <Text className="text-3xl text-accent mb-1" style={{ fontFamily: 'Fredoka-SemiBold' }}>
                  0
                </Text>
                <Text
                  className={`text-sm text-center capitalize ${
                    isDark ? 'text-neutral' : 'text-primary'
                  }`}
                  style={{ fontFamily: 'Montserrat-SemiBold' }}
                >
                  Signs Learned
                </Text>
              </View>

              <View
                className={`w-[48%] rounded-2xl p-4 mb-4 items-center border border-accent shadow-md ${
                  isDark ? 'bg-darksurface' : 'bg-white'
                }`}
              >
                <Text className="text-3xl text-accent mb-1" style={{ fontFamily: 'Fredoka-SemiBold' }}>
                  {userStats.days_streak}
                </Text>
                <Text
                  className={`text-sm text-center capitalize ${
                    isDark ? 'text-neutral' : 'text-primary'
                  }`}
                  style={{ fontFamily: 'Montserrat-SemiBold' }}
                >
                  Days Streak
                </Text>
              </View>

              <View
                className={`w-[48%] rounded-2xl p-4 mb-4 items-center border border-accent shadow-md ${
                  isDark ? 'bg-darksurface' : 'bg-white'
                }`}
              >
                <Text className="text-3xl text-accent mb-1" style={{ fontFamily: 'Fredoka-SemiBold' }}>
                  {userStats.practice_hours.toFixed(1)}
                </Text>
                <Text
                  className={`text-sm text-center capitalize ${
                    isDark ? 'text-neutral' : 'text-primary'
                  }`}
                  style={{ fontFamily: 'Montserrat-SemiBold' }}
                >
                  Practice Hours
                </Text>
              </View>
            </View>

            {/* ⚙️ Settings - MODIFIED TO ONLY INCLUDE DARK MODE */}
            <Text
              className={`text-lg mb-3 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Settings
            </Text>

            <View
              className={`rounded-2xl p-4 shadow-md mb-8 border border-accent ${
                isDark ? 'bg-darksurface' : 'bg-white'
              }`}
            >
              {/* Dark Mode Switch (Retained) */}
              <View
                  className={`flex-row items-center justify-between py-3`} 
              >
                  <View className="flex-row items-center">
                      <MaterialIcons name="dark-mode" size={24} color="#FF6B00" style={{ marginRight: 10 }} />
                      <Text
                          className={`text-base ${isDark ? 'text-secondary' : 'text-primary'}`}
                          style={{ fontFamily: 'Fredoka-Regular' }}
                      >
                          Dark Mode
                      </Text>
                  </View>
                  <Switch
                      trackColor={{ false: '#d1d5db', true: '#FF6B00' }}
                      thumbColor={isDark ? '#fff' : '#f4f3f4'}
                      onValueChange={toggleTheme}
                      value={isDark}
                  />
              </View>
              {/* NOTE: buttonSettings.map loop has been removed */}
            </View>

            <Text
              className={`text-lg mb-3 mt-8 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              About
            </Text>

            <View
              className={`rounded-2xl p-4 shadow-md border border-accent mb-10 ${
                isDark ? 'bg-darksurface' : 'bg-white'
              }`}
            >
              <Text
                className={`text-base mb-4 ${isDark ? 'text-neutral' : 'text-primary'}`}
                style={{ fontFamily: 'Montserrat-SemiBold', lineHeight: 22 }}
              >
                Gesturix is a modern sign language learning companion that helps you
                learn, translate, and communicate using interactive tools and real-time
                sign recognition. Built with love and purpose to make learning sign
                language accessible for everyone.
              </Text>

              <Text
                className={`text-sm mb-2 ${isDark ? 'text-secondary' : 'text-primary'}`}
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                Developed by:
              </Text>

              {[
                { name: 'John Michael A. Nave', url: 'https://github.com/GoldenSnek' },
                { name: 'Jesnar T. Tindogan', url: 'https://github.com/Jasner13' },
                { name: 'John Michael B. Villamor', url: 'https://github.com/Villamormike' },
              ].map((dev, idx) => (
                <TouchableOpacity
                  key={idx}
                  onPress={() => openGitHub(dev.url)}
                  className="active:opacity-70"
                >
                  <Text
                    className="text-accent text-base underline mb-1"
                    style={{ fontFamily: 'Fredoka-SemiBold' }}
                  >
                    {dev.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            {/* 👤 Account */}
            <Text
              className={`text-lg mb-3 ${isDark ? 'text-secondary' : 'text-primary'}`}
              style={{ fontFamily: 'Audiowide-Regular' }}
            >
              Account
            </Text>

            <View
              className={`rounded-2xl p-4 shadow-md border border-accent ${
                isDark ? 'bg-darksurface' : 'bg-white'
              }`}
            >
              {/* Exit App */}
              <TouchableOpacity
                onPress={() => BackHandler.exitApp()}
                className="flex-row items-center justify-between py-2 border-b border-orange-400"
              >
                <Text
                  className="text-base text-yellow-500"
                  style={{ fontFamily: 'Fredoka-SemiBold' }}
                >
                  Exit App
                </Text>
                <AntDesign name="closecircleo" size={22} color="#efc144ff" />
              </TouchableOpacity>

              {/* Sign Out */}
              <TouchableOpacity
                onPress={handleSignOut}
                className="flex-row items-center justify-between py-2"
              >
                <Text
                  className="text-base text-red-500"
                  style={{ fontFamily: 'Fredoka-SemiBold' }}
                >
                  Sign Out
                </Text>
                <AntDesign name="logout" size={22} color="#ef4444" />
              </TouchableOpacity>
            </View>
            
          </ScrollView>
        </View>
      </ImageBackground>
    </View>
  );
};

export default Profile;