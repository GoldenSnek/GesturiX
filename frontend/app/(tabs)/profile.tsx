import React, { useState } from 'react';
import { View, Text, Switch, TouchableOpacity, ScrollView, Image } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons, AntDesign } from '@expo/vector-icons';
import { router } from 'expo-router';

const Profile = () => {
    const insets = useSafeAreaInsets();

    const [isSoundEffectsEnabled, setSoundEffectsEnabled] = useState(false);
    const [isVibrationEnabled, setVibrationEnabled] = useState(false);
    const [isDarkModeEnabled, setDarkModeEnabled] = useState(false);
    const [areNotificationsEnabled, setNotificationsEnabled] = useState(false);

    const toggleSoundEffects = () => setSoundEffectsEnabled(previousState => !previousState);
    const toggleVibration = () => setVibrationEnabled(previousState => !previousState);
    const toggleDarkMode = () => setDarkModeEnabled(previousState => !previousState);
    const toggleNotifications = () => setNotificationsEnabled(previousState => !previousState);

    // Dummy data for progress and achievements
    const progressData = {
        lessonsCompleted: 25,
        signsLearned: 180,
        daysStreak: 7,
        practiceHours: 12.5,
    };

    return (
        <View className="flex-1 bg-gray-100" style={{ paddingTop: insets.top }}>
            {/* Header */}
            <LinearGradient
                colors={['#FF6B00', '#FFAB7B']}
                className="items-center py-2 px-4 flex-row justify-between"
            >
                <TouchableOpacity onPress={() => router.back()}>
                    <MaterialIcons name="arrow-back" size={24} color="primary" />
                </TouchableOpacity>
                <View className="flex-row items-center">
                    <Image
                        source={require('../../assets/images/CatPFP.jpg')}
                        className="w-20 h-20 rounded-full mr-4"
                    />
                    <View>
                        <Text className="text-primary text-xl font-bold">Mr. Pussy Cat</Text>
                        <Text className="text-primary text-sm">pussycat@gmail.com</Text>
                        <Text className="text-primary text-sm">Member since September 11, 2001</Text>
                    </View>
                </View>
                <View style={{ width: 24 }} />
            </LinearGradient>

            <ScrollView className="flex-1 p-4">
                {/* Your Progress Section */}
                <Text className="text-xl font-bold text-gray-800 mb-4">Your Progress</Text>
                <View className="flex-row flex-wrap justify-between mb-8">
                    <View className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center">
                        <Text className="text-3xl font-bold text-[#FF6B00]">{progressData.lessonsCompleted}</Text>
                        <Text className="text-sm text-gray-500 text-center">Lessons Completed</Text>
                    </View>
                    <View className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center">
                        <Text className="text-3xl font-bold text-[#FF6B00]">{progressData.signsLearned}</Text>
                        <Text className="text-sm text-gray-500 text-center">Signs Learned</Text>
                    </View>
                    <View className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center">
                        <Text className="text-3xl font-bold text-[#FF6B00]">{progressData.daysStreak}</Text>
                        <Text className="text-sm text-gray-500 text-center">Days Streak</Text>
                    </View>
                    <View className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center">
                        <Text className="text-3xl font-bold text-[#FF6B00]">{progressData.practiceHours}</Text>
                        <Text className="text-sm text-gray-500 text-center">Practice Hours</Text>
                    </View>
                </View>

                {/* Achievements Section */}
                <Text className="text-xl font-bold text-gray-800 mb-4">Achievements</Text>
                <View className="flex-row flex-wrap justify-between mb-8">
                    <View className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center">
                        <MaterialIcons name="star-border" size={40} color="#FF6B00" />
                        <Text className="text-lg font-bold text-gray-800">First Steps</Text>
                        <Text className="text-sm text-gray-500 text-center">Complete your first lesson</Text>
                    </View>
                    <View className="w-[48%] bg-white rounded-xl p-4 shadow-md mb-4 items-center">
                        <MaterialIcons name="star-border" size={40} color="#FF6B00" />
                        <Text className="text-lg font-bold text-gray-800">Letter Master</Text>
                        <Text className="text-sm text-gray-500 text-center">Learn all 26 letters</Text>
                    </View>
                    <View className="w-[48%] bg-gray-300 rounded-xl p-4 shadow-md mb-4 items-center">
                        <MaterialIcons name="lock-outline" size={40} color="#666" />
                        <Text className="text-lg font-bold text-gray-600">Number Wizard</Text>
                        <Text className="text-sm text-gray-500 text-center">Lessons Completed</Text>
                    </View>
                    <View className="w-[48%] bg-gray-300 rounded-xl p-4 shadow-md mb-4 items-center">
                        <MaterialIcons name="lock-outline" size={40} color="#666" />
                        <Text className="text-lg font-bold text-gray-600">Phrase Expert</Text>
                        <Text className="text-sm text-gray-500 text-center">Signs Learned</Text>
                    </View>
                </View>

                <TouchableOpacity className="flex-row items-center justify-center mb-8">
                    <Text className="text-sm text-gray-500 mr-2">See More</Text>
                    <MaterialIcons name="expand-more" size={20} color="#666" />
                </TouchableOpacity>

                {/* Settings Section */}
                <Text className="text-lg font-bold text-gray-800 mb-4">Settings</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    {/* Sound Effects Toggle */}
                    <View className="flex-row items-center justify-between py-2 border-b border-gray-200">
                        <Text className="text-base font-medium text-gray-800">Sound Effects</Text>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isSoundEffectsEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleSoundEffects}
                            value={isSoundEffectsEnabled}
                        />
                    </View>
                    {/* Vibration Toggle */}
                    <View className="flex-row items-center justify-between py-2 border-b border-gray-200">
                        <Text className="text-base font-medium text-gray-800">Vibration</Text>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isVibrationEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleVibration}
                            value={isVibrationEnabled}
                        />
                    </View>
                    {/* Notifications Toggle */}
                    <View className="flex-row items-center justify-between py-2 border-b border-gray-200">
                        <Text className="text-base font-medium text-gray-800">Notifications</Text>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={areNotificationsEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleNotifications}
                            value={areNotificationsEnabled}
                        />
                    </View>
                    {/* Dark Mode Toggle */}
                    <View className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-medium text-gray-800">Dark Mode</Text>
                        <Switch
                            trackColor={{ false: "#E5E7EB", true: "#FF6B00" }}
                            thumbColor={isDarkModeEnabled ? "#fff" : "#f4f3f4"}
                            ios_backgroundColor="#E5E7EB"
                            onValueChange={toggleDarkMode}
                            value={isDarkModeEnabled}
                        />
                    </View>
                </View>
                
                {/* Support Section */}
                <Text className="text-lg font-bold text-gray-800 mb-4">Support</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    <TouchableOpacity className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-medium text-gray-800">Help Center</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#666" />
                    </TouchableOpacity>
                    <View className="border-b border-gray-200 my-2" />
                    <TouchableOpacity className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-medium text-gray-800">Send Feedback</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#666" />
                    </TouchableOpacity>
                    <View className="border-b border-gray-200 my-2" />
                    <TouchableOpacity className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-medium text-gray-800">About GesturiX</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#666" />
                    </TouchableOpacity>
                </View>

                {/* Account Section */}
                <Text className="text-lg font-bold text-gray-800 mb-4">Account</Text>
                <View className="bg-white rounded-xl p-4 shadow-md mb-6">
                    <TouchableOpacity className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-medium text-gray-800">Edit Profile</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#666" />
                    </TouchableOpacity>
                    <View className="border-b border-gray-200 my-2" />
                    <TouchableOpacity className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-medium text-gray-800">Privacy & Security</Text>
                        <MaterialIcons name="chevron-right" size={24} color="#666" />
                    </TouchableOpacity>
                    <View className="border-b border-gray-200 my-2" />
                    <TouchableOpacity onPress={() => router.replace('/(stack)/LandingPage')} className="flex-row items-center justify-between py-2">
                        <Text className="text-base font-bold text-red-500">Sign Out</Text>
                        <AntDesign name="logout" size={24} color="#ef4444" />
                    </TouchableOpacity>
                </View>

            </ScrollView>
        </View>
    );
};

export default Profile;