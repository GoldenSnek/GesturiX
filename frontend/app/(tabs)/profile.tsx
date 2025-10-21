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
  Pressable,
  Modal,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { MaterialIcons, AntDesign } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { supabase } from '../../src/supabaseClient';
import AppHeaderProfile from '../../components/AppHeaderProfile';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { Buffer } from 'buffer';
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
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [showPhotoModal, setShowPhotoModal] = useState(false);
  const [showPhotoOptions, setShowPhotoOptions] = useState(false);

  const [isEditingUsername, setIsEditingUsername] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [usernameLoading, setUsernameLoading] = useState(false);

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

  const router = useRouter();

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
    router.replace('/(stack)/LandingPage');
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

  const handleViewPhoto = () => {
    setShowPhotoOptions(false);
    setShowPhotoModal(true);
  };

  const handleChangePhoto = async () => {
    setShowPhotoOptions(false);
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
      <View className="flex-1 items-center justify-center bg-secondary">
        <ActivityIndicator size="large" color="#FF6B00" />
      </View>
    );
  }

  if (!profile) {
    return (
      <View className="flex-1 items-center justify-center bg-secondary">
        <Text
          className="text-primary"
          style={{ fontFamily: 'Montserrat-SemiBold' }}
        >
          No profile data found.
        </Text>
      </View>
    );
  }

  return (
    <View
      className="flex-1 bg-secondary"
      style={{ paddingTop: insets.top }}
    >
      <AppHeaderProfile />

      <ScrollView
        {...panResponder.panHandlers}
        className="flex-1 p-4"
        contentContainerStyle={{ paddingBottom: 150 }}
        keyboardShouldPersistTaps="handled"
      >
        {/* 🧍 Profile Photo + Username */}
        <View className="items-center justify-center my-10">
          <View className="relative items-center mb-3">
            <Image
              source={{
                uri:
                  photoUrl ||
                  'https://ui-avatars.com/api/?name=User&background=999&color=fff',
              }}
              style={{
                width: 180,
                height: 180,
                borderRadius: 90,
                backgroundColor: '#eee',
              }}
            />
            <TouchableOpacity
              className="absolute right-0 bottom-4 bg-white rounded-full p-2 shadow-md active:opacity-80"
              onPress={() => setShowPhotoOptions(true)}
            >
              <MaterialIcons name="edit" size={24} color="#FF6B00" />
            </TouchableOpacity>
          </View>

          {/* ✏️ Username */}
          <View className="items-center w-[240px]">
            {!isEditingUsername ? (
              <TouchableOpacity
                onPress={() => setIsEditingUsername(true)}
                activeOpacity={0.8}
                className="flex-row items-center justify-center"
              >
                <Text
                  className="text-2xl text-primary mr-2"
                  style={{ fontFamily: 'Fredoka-SemiBold' }}
                >
                  {profile.username}
                </Text>
                <MaterialIcons name="edit" size={20} color="#FF6B00" />
              </TouchableOpacity>
            ) : (
              <View className="flex-row items-center justify-center">
                <TextInput
                  value={newUsername}
                  onChangeText={setNewUsername}
                  editable={!usernameLoading}
                  className="border-b border-accent text-center text-primary text-lg flex-1"
                  style={{ fontFamily: 'Montserrat-SemiBold' }}
                  autoCapitalize="none"
                />
                <TouchableOpacity
                  onPress={handleSaveUsername}
                  disabled={usernameLoading}
                  className="ml-3"
                >
                  <Text
                    className="text-accent"
                    style={{ fontFamily: 'Fredoka-SemiBold' }}
                  >
                    Save
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => setIsEditingUsername(false)}
                  disabled={usernameLoading}
                  className="ml-3"
                >
                  <Text
                    className="text-neutral"
                    style={{ fontFamily: 'Montserrat-SemiBold' }}
                  >
                    Cancel
                  </Text>
                </TouchableOpacity>
              </View>
            )}
          </View>
        </View>

        {/* 🏆 Progress Section */}
        <Text
          className="text-xl text-primary mb-3"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Your Progress
        </Text>

        <View className="flex-row flex-wrap justify-between mb-8">
          {Object.entries(progressData).map(([key, value]) => (
            <View
              key={key}
              className="w-[48%] bg-white rounded-2xl p-4 shadow-md mb-4 items-center border border-accent"
            >
              <Text
                className="text-3xl text-accent mb-1"
                style={{ fontFamily: 'Fredoka-SemiBold' }}
              >
                {value}
              </Text>
              <Text
                className="text-sm text-neutral text-center capitalize"
                style={{ fontFamily: 'Montserrat-SemiBold' }}
              >
                {key.replace(/([A-Z])/g, ' $1')}
              </Text>
            </View>
          ))}
        </View>

        {/* ⚙️ Settings */}
        <Text
          className="text-lg text-primary mb-3"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Settings
        </Text>

        <View className="bg-white rounded-2xl p-4 shadow-md mb-8 border border-accent">
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
              className={`flex-row items-center justify-between py-3 ${
                idx !== 3 ? 'border-b border-accent' : ''
              }`}
            >
              <Text
                className="text-base text-primary"
                style={{ fontFamily: 'Fredoka-Regular' }}
              >
                {label}
              </Text>
              <Switch
                trackColor={{ false: '#E5E7EB', true: '#FF6B00' }}
                thumbColor={value ? '#fff' : '#f4f3f4'}
                onValueChange={() => setter(!value)}
                value={value}
              />
            </View>
          ))}
        </View>

        {/* 👤 Account */}
        <Text
          className="text-lg text-primary mb-3"
          style={{ fontFamily: 'Audiowide-Regular' }}
        >
          Account
        </Text>

        <View className="bg-white rounded-2xl p-4 shadow-md border border-accent">
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
  );
};

export default Profile;