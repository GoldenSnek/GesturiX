import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  Switch,
  TouchableOpacity,
  ScrollView,
  Image,
  ActivityIndicator,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons, AntDesign } from '@expo/vector-icons';
import { router } from 'expo-router';
import { supabase } from '../../src/supabaseClient';

// ✅ Updated TypeScript interface to include photo_url
interface ProfileData {
  id: string;
  username: string; // <- use username instead of display_name
  email: string;
  created_at: string;
  photo_url?: string | null;
}


const Profile = () => {
  const insets = useSafeAreaInsets();

  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null); // <-- store generated URL
  const [loading, setLoading] = useState(true);

  const [isSoundEffectsEnabled, setSoundEffectsEnabled] = useState(false);
  const [isVibrationEnabled, setVibrationEnabled] = useState(false);
  const [isDarkModeEnabled, setDarkModeEnabled] = useState(false);
  const [areNotificationsEnabled, setNotificationsEnabled] = useState(false);

  const progressData = {
    lessonsCompleted: 25,
    signsLearned: 180,
    daysStreak: 7,
    practiceHours: 12.5,
  };

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

        // 🔹 Generate public URL for photo if exists
        if (data.photo_url) {
          const { data: urlData } = supabase
  .storage
  .from('avatars')
  .getPublicUrl(data.photo_url);

if (urlData?.publicUrl) {
  setPhotoUrl(urlData.publicUrl);
} else {
  console.error('Error getting public URL.');
  setPhotoUrl(null);
}


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
    router.replace('/(stack)/LandingPage');
  };

  if (loading) {
    return (
      <View className="flex-1 items-center justify-center bg-gray-100">
        <ActivityIndicator size="large" color="#FF6B00" />
      </View>
    );
  }

  if (!profile) {
    return (
      <View className="flex-1 items-center justify-center bg-gray-100">
        <Text className="text-gray-600">No profile data found.</Text>
      </View>
    );
  }

  return (
    <View className="flex-1 bg-gray-100" style={{ paddingTop: insets.top }}>
      {/* Header */}
      <LinearGradient
        colors={['#FF6B00', '#FFAB7B']}
        className="items-center py-2 px-4 flex-row justify-between"
      >
        <TouchableOpacity onPress={() => router.back()}>
          <MaterialIcons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>

        <View className="flex-row items-center">
          <Image
            source={
              photoUrl
                ? { uri: photoUrl }
                : require('../../assets/images/CatPFP.jpg')
            }
            className="w-20 h-20 rounded-full mr-4"
          />
          <View>
            <Text className="text-white text-xl font-bold">
              {profile.username || 'No Name'}
            </Text>
            <Text className="text-white text-sm">{profile.email}</Text>
            <Text className="text-white text-sm">
              Member since {new Date(profile.created_at).toLocaleDateString()}
            </Text>
          </View>
        </View>

        <View style={{ width: 24 }} />
      </LinearGradient>

      <ScrollView className="flex-1 p-4">
        {/* Your Progress Section */}
        <Text className="text-xl font-bold text-gray-800 mb-4">Your Progress</Text>
        <View className="flex-row flex-wrap justify-between mb-8">
          {Object.entries(progressData).map(([key, value]) => (
            <View
              key={key}
              className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center"
            >
              <Text className="text-3xl font-bold text-[#FF6B00]">{value}</Text>
              <Text className="text-sm text-gray-500 text-center capitalize">
                {key.replace(/([A-Z])/g, ' $1')}
              </Text>
            </View>
          ))}
        </View>

        {/* Settings Section */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Settings</Text>
        <View className="bg-white rounded-xl p-4 shadow-md mb-6">
          {(
            [
              ['Sound Effects', isSoundEffectsEnabled, setSoundEffectsEnabled],
              ['Vibration', isVibrationEnabled, setVibrationEnabled],
              ['Notifications', areNotificationsEnabled, setNotificationsEnabled],
              ['Dark Mode', isDarkModeEnabled, setDarkModeEnabled],
            ] as [string, boolean, React.Dispatch<React.SetStateAction<boolean>>][]
          ).map(([label, value, setter], idx) => (
            <View
              key={idx}
              className={`flex-row items-center justify-between py-2 ${
                idx !== 3 ? 'border-b border-gray-200' : ''
              }`}
            >
              <Text className="text-base font-medium text-gray-800">{label}</Text>
              <Switch
                trackColor={{ false: '#E5E7EB', true: '#FF6B00' }}
                thumbColor={value ? '#fff' : '#f4f3f4'}
                onValueChange={() => setter(!value)}
                value={value}
              />
            </View>
          ))}
        </View>

        {/* Account Section */}
        <Text className="text-lg font-bold text-gray-800 mb-4">Account</Text>
        <View className="bg-white rounded-xl p-4 shadow-md mb-6">
          <TouchableOpacity
            onPress={handleSignOut}
            className="flex-row items-center justify-between py-2"
          >
            <Text className="text-base font-bold text-red-500">Sign Out</Text>
            <AntDesign name="logout" size={24} color="#ef4444" />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

export default Profile;
