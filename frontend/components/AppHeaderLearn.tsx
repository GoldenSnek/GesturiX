// File: ../../components/AppHeaderLearn.tsx
import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

interface AppHeaderLearnProps {
  title: string;
  completedCount: number;
  totalCount: number;
}

const AppHeaderLearn: React.FC<AppHeaderLearnProps> = ({ title, completedCount, totalCount }) => {
  const router = useRouter();
  const pct = totalCount > 0 ? Math.min(100, Math.round((completedCount / totalCount) * 100)) : 0;

  return (
    <View style={{ backgroundColor: 'transparent' }}>
      {/* Top gradient with back button and title */}
      <LinearGradient
        colors={['#FF6B00', '#FFAB7B']}
        className="py-3 px-4 flex-row items-center justify-between"
      >
        {/* Back button */}
        <TouchableOpacity onPress={() => router.back()} className="p-1">
          <Ionicons name="arrow-back" size={24} color="white" />
        </TouchableOpacity>

        {/* Title and progress text centered */}
        <View className="flex-1 items-center -ml-8">
          <Text className="text-primary text-lg font-fredoka-semibold">{title}</Text>
          <Text className="text-primary text-xs mt-1 font-fredoka">
            {completedCount}/{totalCount} completed
          </Text>
        </View>

        {/* Spacer to balance layout */}
        <View style={{ width: 24 }} />
      </LinearGradient>

      {/* Thin progress bar (white background with orange fill) */}
      <View className="w-full bg-white rounded-full" style={{ height: 6 }}>
        <View
          style={{ width: `${pct}%`, height: 6 }}
          className="bg-[#FF6B00] rounded-full"
        />
      </View>

      {/* Bottom fade gradient */}
      <LinearGradient
        colors={[
          'rgba(255, 171, 123, 1.0)',
          'rgba(255, 171, 123, 0.0)',
        ]}
        start={{ x: 0.5, y: 0.0 }}
        end={{ x: 0.5, y: 1.0 }}
        className="items-center py-2"
      />
    </View>
  );
};

export default AppHeaderLearn;